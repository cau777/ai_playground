use std::error::Error;
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
    memory::allocator::StandardMemoryAllocator,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
    VulkanLibrary,
};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::Queue;
use vulkano::shader::{ShaderCreationError, ShaderModule, SpecializationConstants};
use crate::utils::{ArrayF};

pub type GlobalGpu = Arc<GpuData>;

pub struct GpuData {
    device: Arc<PhysicalDevice>,
    create_info: DeviceCreateInfo,
    // cache_data: Arc<RwLock<Vec<u8>>>,
}

impl GpuData {
    pub fn new_global() -> RunnerResult<GlobalGpu> {
        Self::new().map(|o| Arc::new(o))
    }

    fn new() -> RunnerResult<Self> {
        let library = VulkanLibrary::new().unwrap();
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

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        Ok(Self {
            device: physical_device,
            create_info: DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
            // cache_data: Arc::new(RwLock::new(Vec::new())),
        })
    }
}

type RunnerResult<T> = Result<T, Box<dyn Error>>;

pub struct ShaderRunner {
    descriptors: Vec<WriteDescriptorSet>,
    device: Arc<Device>,
    pipeline: Arc<ComputePipeline>,
    memory_alloc: StandardMemoryAllocator,
    cmd_alloc: StandardCommandBufferAllocator,
    descriptor_set_alloc: StandardDescriptorSetAllocator,
    queues: Box<dyn ExactSizeIterator<Item=Arc<Queue>>>,
    gpu: GlobalGpu,
    // cache: Arc<PipelineCache>,
}

impl ShaderRunner {
    pub fn new(gpu: GlobalGpu, func: impl FnOnce(Arc<Device>) -> Result<Arc<ShaderModule>, ShaderCreationError>,
               entrypoint: &str, constants: &impl SpecializationConstants) -> RunnerResult<ShaderRunner> {
        // Now initializing the device.
        let (device, queues) = Device::new(gpu.device.clone(), gpu.create_info.clone())?;

        // let cache = unsafe {
        //     PipelineCache::with_data(device.clone(), &gpu.cache_data.read().unwrap())?
        // };
        let pipeline = {
            let shader = func(device.clone())?;
            let entry = shader.entry_point(entrypoint)
                .ok_or_else(|| format!("Can't find function {} on shader", entrypoint))?;

            ComputePipeline::new(
                device.clone(),
                entry,
                constants,
                // Some(cache.clone()),
                None,
                |_| {},
            )?
        };

        Ok(ShaderRunner {
            memory_alloc: StandardMemoryAllocator::new_default(device.clone()),
            descriptor_set_alloc: StandardDescriptorSetAllocator::new(device.clone()),
            cmd_alloc: StandardCommandBufferAllocator::new(device.clone(), Default::default()),
            pipeline,
            descriptors: Vec::new(),
            device,
            queues: Box::new(queues),
            gpu,
            // cache,
        })
    }

    pub fn create_buffer_from_array<D: Dimension>(&mut self, array: &ArrayF<D>) -> RunnerResult<Arc<CpuAccessibleBuffer<[f32]>>> {
        let buffer = CpuAccessibleBuffer::from_iter(
            &self.memory_alloc,
            BufferUsage {
                storage_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            array.iter().copied(),
        )?;
        self.descriptors.push(WriteDescriptorSet::buffer(self.descriptors.len() as u32, buffer.clone()));
        Ok(buffer)
    }

    pub fn execute(&mut self, total_times: [u32; 3], block_size: [u32; 3]) -> RunnerResult<()> {
        if (0..3).into_iter().any(|o| total_times[o] % block_size[o] != 0) {
            Err("Invalid groups: not divisible")?; // TODO: better error
        }

        let group_counts = [total_times[0] / block_size[0], total_times[1] / block_size[1], total_times[2] / block_size[2]];
        if group_counts.iter().copied().any(|o| o == 0) {
            Err("Invalid groups: count = 0")?; // TODO: better error
        }

        let layouts = self.pipeline.layout().set_layouts();
        let layout = if layouts.len() == 0 { Err("No layouts found") } else { Ok(&layouts[0]) };
        let queue = self.queues.next().expect("Should create queues");

        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_alloc,
            layout?.clone(),
            self.descriptors.drain(0..).into_iter(),
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.cmd_alloc,
            queue.queue_family_index(),
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

        let command_buffer = builder.build()?;
        let future = sync::now(self.device.clone())
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None)?;

        // let data = self.cache.get_data()?;
        // let mut global_cache = self.gpu.cache_data.write().unwrap();
        // if data.len() > 1_000_000 {
        //     println!("Cache size = {:?}KB", data.len() / 1_000);
        // }
        // *global_cache = data;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test() {}
}