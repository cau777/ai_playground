use std::sync::{Arc};
use ndarray::Dimension;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
    VulkanLibrary,
};
use vulkano::command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer};
use vulkano::device::Queue;
use vulkano::memory::allocator::FastMemoryAllocator;
use vulkano::pipeline::cache::PipelineCache;
use vulkano::shader::{ShaderCreationError, ShaderModule, SpecializationConstants};
use vulkano::sync::{FenceSignalFuture, NowFuture};
use crate::utils::{ArrayF, GenericResult};

pub type GlobalGpu = Arc<GpuData>;

pub struct GpuData {
    device: Arc<Device>,
    // Queues are the equivalent of CPU threads
    queue: Arc<Queue>,
    descriptor_alloc: StandardDescriptorSetAllocator,
    cmd_alloc: StandardCommandBufferAllocator,
    cache: Option<Arc<PipelineCache>>,
}

impl GpuData {
    pub fn new_global() -> GenericResult<GlobalGpu> {
        Self::new().map(Arc::new)
    }

    fn new() -> GenericResult<Self> {
        let library = VulkanLibrary::new()?;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
                enumerate_portability: true,
                ..Default::default()
            },
        )?;

        // Choose which physical device to use.
        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()?
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
                // that supports compute operations.
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.compute)
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }).ok_or_else(|| "Can't find suitable device".to_owned())?;

        // println!(
        //     "Using device: {} (type: {:?})",
        //     physical_device.properties().device_name,
        //     physical_device.properties().device_type
        // );

        let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        })?;

        Ok(Self {
            queue: queues.next().ok_or("Should create 1 queue")?,
            // memory_alloc: StandardMemoryAllocator::new_default(device.clone()),
            descriptor_alloc: StandardDescriptorSetAllocator::new(device.clone()),
            cmd_alloc: StandardCommandBufferAllocator::new(device.clone(), Default::default()),
            cache: PipelineCache::empty(device.clone()).map_err(|e| println!("{:?}", e)).ok(),
            device,
        })
    }
}

pub struct ShaderRunner {
    descriptors: Vec<WriteDescriptorSet>,
    pipeline: Arc<ComputePipeline>,
    gpu: GlobalGpu,
    memory_alloc: FastMemoryAllocator,
    buffers: Vec<Arc<CpuAccessibleBuffer<[f32]>>>,
}

impl ShaderRunner {
    pub fn new(gpu: GlobalGpu, func: impl FnOnce(Arc<Device>) -> Result<Arc<ShaderModule>, ShaderCreationError>,
               entrypoint: &str, constants: &impl SpecializationConstants) -> GenericResult<ShaderRunner> {
        let pipeline = {
            let shader = func(gpu.device.clone())?;
            let entry = shader.entry_point(entrypoint)
                .ok_or_else(|| format!("Can't find function {} on shader", entrypoint))?;

            ComputePipeline::new(
                gpu.device.clone(),
                entry,
                constants,
                gpu.cache.clone(),
                |_| {},
            )?
        };

        Ok(ShaderRunner {
            pipeline,
            descriptors: Vec::new(),
            memory_alloc: FastMemoryAllocator::new_default(gpu.device.clone()),
            gpu,
            buffers: Vec::new(),
        })
    }

    /// Create a buffer that will be used to transfer data to the GPU
    pub fn create_read_buffer<D: Dimension>(&mut self, array: &ArrayF<D>) -> GenericResult<()> {
        let buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_alloc,
            BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                ..BufferUsage::empty()
            },
            false,
            array.iter().copied(),
        )?;
        self.add_buffer(buffer);
        Ok(())

        // let buffer = DeviceLocalBuffer::from_iter(
        //     &self.gpu.memory_alloc,
        //     array.iter().copied(),
        //     BufferUsage {
        //         storage_buffer: true,
        //         transfer_dst: true,
        //         ..BufferUsage::empty()
        //     },
        //     &mut AutoCommandBufferBuilder::primary(
        //         &self.gpu.cmd_alloc,
        //         self.gpu.queue.queue_family_index(),
        //         CommandBufferUsage::OneTimeSubmit,
        //     )?,
        // )?;
        // self.add_buffer(buffer);
        // Ok(())
    }

    /// Create an empty buffer that will be used to transfer data from the GPU
    pub fn create_write_buffer_uninit(&mut self, len: usize) -> GenericResult<Arc<CpuAccessibleBuffer<[f32]>>> {
        let buffer = unsafe {
            CpuAccessibleBuffer::<[f32]>::uninitialized_array(
                &self.memory_alloc,
                len as vulkano::DeviceSize,
                BufferUsage {
                    storage_buffer: true,
                    transfer_src: true,
                    ..BufferUsage::empty()
                },
                false,
            )
        }?;
        self.add_buffer(buffer.clone());
        Ok(buffer)
    }

    /// Create an empty buffer that will be used to transfer data from the GPU
    pub fn create_write_buffer(&mut self, len: usize) -> GenericResult<Arc<CpuAccessibleBuffer<[f32]>>> {
        let buffer =
            CpuAccessibleBuffer::from_iter(
                &self.memory_alloc,
                BufferUsage {
                    storage_buffer: true,
                    transfer_src: true,
                    ..BufferUsage::empty()
                },
                false,
                (0..len).into_iter().map(|_| 0.0),
            )?;
        self.add_buffer(buffer.clone());
        Ok(buffer)
    }

    /// Create a generic buffer that supports read and write operations
    pub fn create_buffer<D: Dimension>(&mut self, array: &ArrayF<D>) -> GenericResult<Arc<CpuAccessibleBuffer<[f32]>>> {
        let buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_alloc,
            BufferUsage {
                storage_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            array.iter().copied(),
        )?;
        self.add_buffer(buffer.clone());
        Ok(buffer)
    }

    pub fn execute(&mut self, total_times: [u32; 3], block_size: [u32; 3]) -> GenericResult<()> {
        if (0..3).into_iter().any(|o| total_times[o] % block_size[o] != 0) {
            Err("Invalid groups: not divisible")?; // TODO: better error
        }

        let group_counts = [total_times[0] / block_size[0], total_times[1] / block_size[1], total_times[2] / block_size[2]];
        if group_counts.iter().copied().any(|o| o == 0) {
            Err("Invalid groups: count = 0")?; // TODO: better error
        }

        let layouts = self.pipeline.layout().set_layouts();
        let layout = layouts.get(0).ok_or("No layouts found")?;

        let set = PersistentDescriptorSet::new(
            &self.gpu.descriptor_alloc,
            layout.clone(),
            self.descriptors.drain(0..),
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.gpu.cmd_alloc,
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // println!("Running {:?} times", group_counts);
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch(group_counts)?;

        let cmd = builder.build()?;
        self.exec_cmd(cmd)?.wait(None)?;

        Ok(())
    }

    fn exec_cmd(&self, cmd: PrimaryAutoCommandBuffer) -> GenericResult<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>> {
        Ok(sync::now(self.gpu.device.clone())
            .then_execute(self.gpu.queue.clone(), cmd)?
            .then_signal_fence_and_flush()?)
    }

    fn add_buffer(&mut self, buffer: Arc<CpuAccessibleBuffer<[f32]>>) {
        self.descriptors.push(WriteDescriptorSet::buffer(self.descriptors.len() as u32, buffer.clone()));
        self.buffers.push(buffer);
    }
}

impl Drop for ShaderRunner {
    /// Necessary to avoid memory leaks
    fn drop(&mut self) {
        self.buffers.drain(0..).for_each(drop);
        self.gpu.cmd_alloc.clear(self.gpu.queue.queue_family_index());
        self.gpu.descriptor_alloc.clear_all();
    }
}