use std::sync::Arc;
use ndarray::Dimension;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};
use vulkano::buffer::{BufferAccess};
use vulkano::command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer};
use vulkano::memory::allocator::{FastMemoryAllocator};
use vulkano::shader::{ShaderCreationError, ShaderModule, SpecializationConstants};
use vulkano::sync::{FenceSignalFuture, NowFuture};
use crate::gpu::gpu_data::GlobalGpu;
use crate::utils::{ArrayF, GenericResult};

pub struct ShaderRunner {
    descriptors: Vec<WriteDescriptorSet>,
    pipeline: Arc<ComputePipeline>,
    gpu: GlobalGpu,
    memory_alloc: FastMemoryAllocator,
    buffers: Vec<Arc<dyn BufferAccess>>,
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
    pub fn create_gpu_only_buffer<D: Dimension>(&mut self, array: &ArrayF<D>) -> GenericResult<()> {
        let buffer = self.gpu.pools.gpu_only_pool.from_iter(array.iter().copied())?;
        self.add_buffer(buffer);
        Ok(())
    }

    /// Create an empty buffer that will be used to transfer data from the GPU
    pub fn create_download_buffer(&mut self, len: usize) -> GenericResult<Arc<CpuAccessibleBuffer<[f32]>>> {
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

    fn create_groups(total_times: [u32; 3], block_size: [u32; 3]) -> Result<[u32; 3], String> {
        for x in 0..3 {
            let total = total_times[x];
            let block = block_size[x];
            if total < block {
                return Err(format!("Invalid groups: {} is smaller than the block size {} in group {}", total, block, x));
            }

            if total % block != 0 {
                return Err(format!("Invalid groups: {} is not divisible by {} in group {}", total, block, x));
            }
        }

        Ok([total_times[0] / block_size[0], total_times[1] / block_size[1], total_times[2] / block_size[2]])
    }

    pub fn execute(&mut self, total_times: [u32; 3], block_size: [u32; 3]) -> GenericResult<()> {
        let group_counts = Self::create_groups(total_times, block_size)?;

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

    fn add_buffer(&mut self, buffer: Arc<dyn BufferAccess>) {
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