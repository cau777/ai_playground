use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, physical::PhysicalDeviceType, Queue, QueueCreateInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::memory::allocator::{FastMemoryAllocator, StandardMemoryAllocator};
use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::sync::{FenceSignalFuture, GpuFuture, NowFuture};
use crate::gpu::pools::Pools;
use crate::gpu::shader_context::{ShaderContextKey, ShaderContext};
use crate::utils::GenericResult;

pub type GlobalGpu = Arc<GpuData>;

pub struct GpuData {
    pub device: Arc<Device>,
    // Queues are the equivalent of CPU threads
    pub queue: Arc<Queue>,
    pub descriptor_alloc: StandardDescriptorSetAllocator,
    pub cmd_alloc: StandardCommandBufferAllocator,
    pub cache: Option<Arc<PipelineCache>>,
    pub std_mem_alloc: Arc<StandardMemoryAllocator>,
    pub fast_mem_alloc: RwLock<FastMemoryAllocator>,
    pub pools: Pools,
    pub contexts: RwLock<HashMap<ShaderContextKey, ShaderContext>>,
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
            }).ok_or_else(|| anyhow::anyhow!("Can't find suitable device"))?;

        let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        })?;

        let memory_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        Ok(Self {
            queue: queues.next().ok_or_else(||anyhow::anyhow!("Should create 1 queue"))?,
            // memory_alloc: StandardMemoryAllocator::new_default(device.clone()),
            descriptor_alloc: StandardDescriptorSetAllocator::new(device.clone()),
            cmd_alloc: StandardCommandBufferAllocator::new(device.clone(), Default::default()),
            cache: PipelineCache::empty(device.clone()).map_err(|e| println!("{:?}", e)).ok(),
            fast_mem_alloc: RwLock::new(FastMemoryAllocator::new_default(device.clone())),
            device,
            pools: Pools::new(memory_alloc.clone()),
            std_mem_alloc: memory_alloc,
            contexts: RwLock::new(HashMap::new()),
        })
    }

    pub fn exec_cmd(&self, cmd: PrimaryAutoCommandBuffer) -> GenericResult<FenceSignalFuture<CommandBufferExecFuture<NowFuture>>> {
        Ok(vulkano::sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cmd)?
            .then_signal_fence_and_flush()?)
    }

    pub fn reset_fast_mem_alloc(&self) -> GenericResult<()> {
        *self.fast_mem_alloc.write().unwrap() = FastMemoryAllocator::new_default(self.device.clone());
        Ok(())
    }
}
