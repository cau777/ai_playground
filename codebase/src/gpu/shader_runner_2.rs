use std::sync::Arc;
use vulkano::buffer::{ CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::shader::{ShaderCreationError, ShaderModule, SpecializationConstants};
use crate::gpu::gpu_data::{GlobalGpu, PipelineObjects};
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::GenericResult;

pub struct PipelineCreateInfo<TConstants: SpecializationConstants, TLoadModule: FnOnce(Arc<Device>) -> LoadModuleResult> {
    pub load_module: TLoadModule,
    pub entry: String,
    pub constants: TConstants,
}

pub struct ShaderRunner2 {
    id: String,
    gpu: GlobalGpu,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

type GpuBuffer = Arc<DeviceLocalBuffer<[f32]>>;
type LoadModuleResult = Result<Arc<ShaderModule>, ShaderCreationError>;

impl ShaderRunner2 {
    pub fn new<TConstants: SpecializationConstants, TLoadModule: FnOnce(Arc<Device>) -> LoadModuleResult>(
        unique_id: String, gpu: GlobalGpu, buffers_lengths: Vec<usize>,
        info: impl FnOnce() -> PipelineCreateInfo<TConstants, TLoadModule>,
    ) -> GenericResult<Self> {
        let should_create = {
            let read = gpu.pipeline_objects.read().unwrap();
            let prev = read.get(&unique_id);
            prev.is_none() || prev.unwrap().buffers_lengths != buffers_lengths
        };

        if should_create {
            let mut write = gpu.pipeline_objects.write().unwrap();
            write.insert(unique_id.clone(), Self::build_objects(&gpu, buffers_lengths, (info)())?);
        }

        let builder = AutoCommandBufferBuilder::primary(
            &gpu.cmd_alloc,
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;


        Ok(Self {
            id: unique_id,
            gpu,
            builder,
        })
    }

    fn build_objects<TConstants: SpecializationConstants, TLoadModule: FnOnce(Arc<Device>) -> LoadModuleResult>
    (gpu: &GlobalGpu, buffers_lengths: Vec<usize>, info: PipelineCreateInfo<TConstants, TLoadModule>) -> GenericResult<PipelineObjects> {
        if buffers_lengths.is_empty() {
            panic!("By convention, all shaders must have an output buffer");
        }
        let pipeline = Self::build_pipeline(gpu, info.load_module, &info.entry, &info.constants)?;

        let mut buffers = Vec::new();
        let mut writes = Vec::new();

        for (index, &len) in buffers_lengths.iter().enumerate() {
            let buffer = DeviceLocalBuffer::<[f32]>::array(
                &gpu.memory_alloc,
                len as u64,
                vulkano::buffer::BufferUsage {
                    storage_buffer: true,
                    transfer_dst: true,
                    transfer_src: true,
                    ..vulkano::buffer::BufferUsage::empty()
                },
                gpu.device.active_queue_family_indices().iter().copied(),
            )?;

            buffers.push(buffer.clone());
            writes.push(WriteDescriptorSet::buffer(index as u32, buffer));
        }

        let layouts = pipeline.layout().set_layouts();
        let layout = layouts.get(0)
            .ok_or_else(|| anyhow::anyhow!("No layouts found"))?;

        let descriptor_set = PersistentDescriptorSet::new(
            &gpu.descriptor_alloc,
            layout.clone(),
            writes.into_iter(),
        )?;

        Ok(PipelineObjects {
            pipeline,
            buffers,
            descriptor_set,
            buffers_lengths
        })
    }

    fn build_pipeline(gpu: &GlobalGpu, load_module: impl FnOnce(Arc<Device>) -> LoadModuleResult,
                      entrypoint: &str, constants: &impl SpecializationConstants) -> GenericResult<Arc<ComputePipeline>> {
        let shader = load_module(gpu.device.clone())?;
        let entry = shader.entry_point(entrypoint)
            .ok_or_else(|| anyhow::anyhow!("Can't find function {} on shader", entrypoint))?;

        Ok(ComputePipeline::new(
            gpu.device.clone(),
            entry,
            constants,
            gpu.cache.clone(),
            |_| {},
        )?)
    }

    pub fn create_input_buffer(&mut self, binding: usize, data: StoredArray) -> GenericResult<()> {
        match data {
            StoredArray::Memory { data } => {
                let read = self.gpu.pipeline_objects.read().unwrap();
                let objects = read.get(&self.id)
                    .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;

                let buffer = CpuAccessibleBuffer::from_iter(
                    &self.gpu.memory_alloc,
                    vulkano::buffer::BufferUsage {
                        transfer_src: true,
                        ..vulkano::buffer::BufferUsage::empty()
                    },
                    false,
                    data.iter().copied(),
                )?;

                self.builder
                    .copy_buffer(CopyBufferInfo::buffers(buffer, objects.buffers[binding].clone()))?;
            }
            StoredArray::GpuLocal { data, .. } => {
                let read = self.gpu.pipeline_objects.read().unwrap();
                let objects = read.get(&self.id)
                    .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;

                self.builder
                    .copy_buffer(CopyBufferInfo::buffers(data, objects.buffers[binding].clone()))?;
            }
        }
        Ok(())
    }

    pub fn execute(self, total_times: [u32; 3], block_size: [u32; 3]) -> GenericResult<GpuBuffer> {
        let group_counts = Self::create_groups(total_times, block_size)?;

        let Self { mut builder, id, gpu, .. } = self;
        let read = gpu.pipeline_objects.read().unwrap();
        let objects = read.get(&id)
            .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;

        builder
            .bind_pipeline_compute(objects.pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Compute,
                objects.pipeline.layout().clone(),
                0,
                objects.descriptor_set.clone(),
            )
            .dispatch(group_counts)?;

        let cmd = builder.build()?;
        gpu.exec_cmd(cmd)?.wait(None)?;

        // Necessary to avoid memory leaks
        gpu.cmd_alloc.clear(gpu.queue.queue_family_index());
        gpu.descriptor_alloc.clear_all();

        Ok(objects.buffers[0].clone())
    }

    fn create_groups(total_times: [u32; 3], block_size: [u32; 3]) -> GenericResult<[u32; 3]> {
        for x in 0..3 {
            let total = total_times[x];
            let block = block_size[x];
            if total < block {
                return Err(anyhow::anyhow!("Invalid groups: {} is smaller than the block size {} in group {}", total, block, x));
            }

            if total % block != 0 {
                return Err(anyhow::anyhow!("Invalid groups: {} is not divisible by {} in group {}", total, block, x));
            }
        }

        Ok([total_times[0] / block_size[0], total_times[1] / block_size[1], total_times[2] / block_size[2]])
    }
}
