use std::sync::Arc;
use vulkano::buffer::{BufferAccess, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::shader::{ShaderCreationError, ShaderModule, SpecializationConstants};
use crate::gpu::gpu_data::GlobalGpu;
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::GenericResult;

pub struct ShaderRunner2 {
    gpu: GlobalGpu,
    pipeline: Arc<ComputePipeline>,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    descriptor_writes: Vec<WriteDescriptorSet>,
    out_shape: Vec<usize>,
    out_buffer: GpuBuffer,
}

type GpuBuffer = Arc<DeviceLocalBuffer<[f32]>>;
type LoadModuleResult = Result<Arc<ShaderModule>, ShaderCreationError>;

impl ShaderRunner2 {
    pub fn new_io(gpu: GlobalGpu, load_module: impl FnOnce(Arc<Device>) -> LoadModuleResult,
                  entrypoint: &str, constants: &impl SpecializationConstants,
                  out_shape: impl Iterator<Item=usize>) -> GenericResult<Self> {
        // TODO: reuse pipeline
        let pipeline = Self::build_pipeline(&gpu, load_module, entrypoint, constants)?;

        let builder = AutoCommandBufferBuilder::primary(
            &gpu.cmd_alloc,
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let out_shape: Vec<_> = out_shape.collect();
        let out_buffer = DeviceLocalBuffer::<[f32]>::array(
            &gpu.memory_alloc,
            out_shape.iter().map(|&o| o as u64).reduce(|a, b| a * b).unwrap(),
            vulkano::buffer::BufferUsage {
                transfer_src: true,
                storage_buffer: true,
                ..vulkano::buffer::BufferUsage::empty()
            },
            gpu.device.active_queue_family_indices().iter().copied(),
        )?;

        Ok(Self {
            gpu,
            pipeline,
            builder,
            descriptor_writes: vec![
                WriteDescriptorSet::buffer(0, out_buffer.clone())
            ],
            out_shape,
            out_buffer,
        })
    }

    pub fn new_inplace(gpu: GlobalGpu, load_module: impl FnOnce(Arc<Device>) -> LoadModuleResult,
                       entrypoint: &str, constants: &impl SpecializationConstants,
                       array: StoredArray) -> GenericResult<Self> {
        // TODO: reuse pipeline
        let pipeline = Self::build_pipeline(&gpu, load_module, entrypoint, constants)?;

        let builder = AutoCommandBufferBuilder::primary(
            &gpu.cmd_alloc,
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let out_shape = match &array {
            StoredArray::Memory { data } => data.shape().iter().copied().collect(),
            StoredArray::GpuLocal { shape, .. } => shape.clone(),
        };
        let out_buffer = array.into_gpu_local(gpu.clone())?;

        Ok(Self {
            gpu,
            pipeline,
            builder,
            descriptor_writes: vec![
                WriteDescriptorSet::buffer(0, out_buffer.clone())
            ],
            out_shape,
            out_buffer,
        })
    }

    fn build_pipeline(gpu: &GlobalGpu, load_module: impl FnOnce(Arc<Device>) -> LoadModuleResult,
                      entrypoint: &str, constants: &impl SpecializationConstants) -> GenericResult<Arc<ComputePipeline>> {
        let shader = load_module(gpu.device.clone())?;
        let entry = shader.entry_point(entrypoint)
            .ok_or_else(|| format!("Can't find function {} on shader", entrypoint))?;

        Ok(ComputePipeline::new(
            gpu.device.clone(),
            entry,
            constants,
            gpu.cache.clone(),
            |_| {},
        )?)
    }

    // pub fn create_gpu_buffer_from_buffer(&mut self, buffer: Arc<DeviceLocalBuffer<[f32]>>)
    //                                      -> GenericResult<Arc<DeviceLocalBuffer<[f32]>>> {
    //     let buffer_copy = DeviceLocalBuffer::from_buffer(
    //         &self.gpu.memory_alloc.clone(),
    //         buffer,
    //         vulkano::buffer::BufferUsage {
    //             transfer_src: true,
    //             transfer_dst: true,
    //             storage_buffer: true,
    //             ..vulkano::buffer::BufferUsage::empty()
    //         },
    //         &mut self.builder,
    //     )?;
    // 
    //     self.add_buffer(buffer_copy.clone());
    //     Ok(buffer_copy)
    // }

    pub fn create_input_buffer(&mut self, data: StoredArray) -> GenericResult<()> {
        match data {
            StoredArray::Memory { data } => {
                let buffer = CpuAccessibleBuffer::from_iter(
                    &self.gpu.memory_alloc.clone(),
                    vulkano::buffer::BufferUsage {
                        transfer_src: true,
                        transfer_dst: true,
                        storage_buffer: true,
                        ..vulkano::buffer::BufferUsage::empty()
                    },
                    true,
                    data.iter().copied(),
                )?;
                self.add_buffer(buffer.clone());
            }
            StoredArray::GpuLocal { data, .. } => {
                self.add_buffer(data);
            }
        }
        Ok(())
    }

    fn add_buffer(&mut self, buffer: Arc<dyn BufferAccess>) {
        self.descriptor_writes.push(WriteDescriptorSet::buffer(self.descriptor_writes.len() as u32, buffer));
    }

    pub fn execute(mut self, total_times: [u32; 3], block_size: [u32; 3]) -> GenericResult<GpuBuffer> {
        let group_counts = Self::create_groups(total_times, block_size)?;

        let layouts = self.pipeline.layout().set_layouts();
        let layout = layouts.get(0).ok_or("No layouts found")?;

        let set = PersistentDescriptorSet::new(
            &self.gpu.descriptor_alloc,
            layout.clone(),
            self.descriptor_writes.drain(0..),
        )?;

        self.builder
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch(group_counts)?;

        let Self { builder, out_buffer, gpu, .. } = self;
        let cmd = builder.build()?;
        gpu.exec_cmd(cmd)?.wait(None)?;

        /// Necessary to avoid memory leaks
        gpu.cmd_alloc.clear(gpu.queue.queue_family_index());
        gpu.descriptor_alloc.clear_all();

        Ok(out_buffer)
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
}
