use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, BufferCopy, CommandBufferUsage, CopyBufferInfo, FillBufferInfo, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::shader::{ShaderCreationError, ShaderModule, SpecializationConstants};
use crate::ArrayDynF;
use crate::gpu::checksum::{BufferChecksumMethod, checksum_slice, CHUNK_SIZE};
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::pipeline_objects::{BufferObject, PipelineObjects};
use crate::gpu::shader_context::{ContextBinding, ShaderContextKey};
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::GenericResult;

pub struct PipelineCreateInfo<TConstants: SpecializationConstants, TLoadModule: FnOnce(Arc<Device>) -> LoadModuleResult> {
    pub load_module: TLoadModule,
    pub entries: Vec<String>,
    pub constants: TConstants,
}

pub struct ShaderRunner2 {
    context: ShaderContextKey,
    gpu: GlobalGpu,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

type GpuBuffer = Arc<DeviceLocalBuffer<[f32]>>;
type LoadModuleResult = Result<Arc<ShaderModule>, ShaderCreationError>;

impl ShaderRunner2 {
    pub fn new(context_key: ShaderContextKey, gpu: GlobalGpu) -> GenericResult<Self> {
        let builder = AutoCommandBufferBuilder::primary(
            &gpu.cmd_alloc,
            gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        Ok(Self {
            context: context_key,
            gpu,
            builder,
        })
    }

    fn build_objects<TConstants: SpecializationConstants, TLoadModule: FnOnce(Arc<Device>) -> LoadModuleResult>
    (gpu: &GlobalGpu, buffers_lengths:&[usize], info: PipelineCreateInfo<TConstants, TLoadModule>) -> GenericResult<PipelineObjects> {
        if buffers_lengths.is_empty() {
            panic!("By convention, all shaders must have an output buffer");
        }
        let mut pipelines= HashMap::new();
        let shader = (info.load_module)(gpu.device.clone())?;
        for entry in &info.entries {
            pipelines.insert(entry.to_owned(), Self::build_pipeline(gpu, &shader, entry, &info.constants)?);
        }
        let any_pipeline = pipelines.values().next().unwrap();

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

            buffers.push(BufferObject {
                buffer: buffer.clone(),
                length: len,
                checksums: vec![],
            });
            writes.push(WriteDescriptorSet::buffer(index as u32, buffer));
        }

        let layouts = any_pipeline.layout().set_layouts();
        let layout = layouts.get(0)
            .ok_or_else(|| anyhow::anyhow!("No layouts found"))?;

        let descriptor_set = PersistentDescriptorSet::new(
            &gpu.descriptor_alloc,
            layout.clone(),
            writes.into_iter(),
        )?;

        Ok(PipelineObjects {
            pipelines,
            buffers,
            descriptor_set,
        })
    }

    fn build_pipeline(gpu: &GlobalGpu, shader: &ShaderModule,
                      entrypoint: &str, constants: &impl SpecializationConstants) -> GenericResult<Arc<ComputePipeline>> {
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

    pub fn update_buffer_with_memory(&mut self, binding: ContextBinding, data: &ArrayDynF,
                                     checksum: BufferChecksumMethod) -> GenericResult<&mut Self> {
        self.update_buffer_with_memory_checked(binding, data, checksum, &mut false)
    }

    pub fn update_buffer_with_memory_checked(&mut self, binding: ContextBinding, data: &ArrayDynF,
                                     checksum: BufferChecksumMethod, changed: &mut bool) -> GenericResult<&mut Self> {
        let buffer = CpuAccessibleBuffer::from_iter(
            &self.gpu.memory_alloc,
            vulkano::buffer::BufferUsage {
                transfer_src: true,
                ..vulkano::buffer::BufferUsage::empty()
            },
            false,
            data.iter().copied(),
        )?;

        let mut write = self.gpu.contexts.write().unwrap();
        let context = write.get_mut(&self.context)
            .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;

        let buffer_obj = &mut context.get_buffer_object_mut(binding)
            .ok_or_else(|| anyhow::anyhow!("Binding {:?} not found", binding))?;

        let copy_info = match checksum {
            BufferChecksumMethod::None => {
                buffer_obj.checksums = vec![];
                Some(CopyBufferInfo::buffers(buffer, buffer_obj.buffer.clone()))
            }
            BufferChecksumMethod::Single => {
                match data.as_slice() {
                    Some(slice) => {
                        let checksum = vec![checksum_slice(slice)];
                        let should_copy = checksum != buffer_obj.checksums;
                        buffer_obj.checksums = checksum;

                        if should_copy {
                            Some(CopyBufferInfo::buffers(buffer, buffer_obj.buffer.clone()))
                        } else {
                            None
                        }
                    }
                    None => {
                        println!("No slice");
                        buffer_obj.checksums = vec![];
                        Some(CopyBufferInfo::buffers(buffer, buffer_obj.buffer.clone()))
                    }
                }
            }
            BufferChecksumMethod::Split => {
                match data.as_slice() {
                    Some(slice) => {
                        let chunk_size = CHUNK_SIZE;
                        let checksums: Vec<_> = slice.chunks(chunk_size).map(checksum_slice).collect();

                        let result = if checksums.len() != buffer_obj.checksums.len() {
                            // If the lengths are different, perform a full copy
                            Some(CopyBufferInfo::buffers(buffer, buffer_obj.buffer.clone()))
                        } else if checksums != buffer_obj.checksums {
                            // If the buffers have some different chunks, copy only the different chunks
                            let mut info = CopyBufferInfo::buffers(buffer, buffer_obj.buffer.clone());
                            info.regions.clear();

                            for (index, (prev, new)) in std::iter::zip(&buffer_obj.checksums, &checksums).enumerate() {
                                if prev != new {
                                    let offset = chunk_size * index;
                                    let copy = BufferCopy {
                                        src_offset: offset as u64,
                                        dst_offset: offset as u64,
                                        size: u64::min(buffer_obj.length - offset as u64, chunk_size as u64),
                                        ..BufferCopy::default()
                                    };
                                    info.regions.push(copy);
                                }
                            }

                            Some(info)
                        } else {
                            // If the buffers are equal
                            None
                        };

                        buffer_obj.checksums = checksums;
                        result
                    }
                    None => {
                        buffer_obj.checksums = vec![];
                        Some(CopyBufferInfo::buffers(buffer, buffer_obj.buffer.clone()))
                    }
                }
            }
        };

        match copy_info {
            Some(copy_info) => {
                self.builder.copy_buffer(copy_info)?;
                *changed = true;
            }
            None => {
                *changed = false;
            }
        };

        drop(write);
        Ok(self)
    }

    pub fn update_buffer_with_buffer(&mut self, binding: ContextBinding, data: GpuBuffer) -> GenericResult<&mut Self> {
        let read = self.gpu.contexts.read().unwrap();
        let context = read.get(&self.context)
            .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;

        let dst_buffer = context.get_buffer(binding)
            .ok_or_else(|| anyhow::anyhow!("Binding {:?} was not found", binding))?;

        self.builder
            .copy_buffer(CopyBufferInfo::buffers(data, dst_buffer))?;

        drop(read);
        Ok(self)
    }

    pub fn update_buffer_with_binding(&mut self, src_binding: ContextBinding, dst_binding: ContextBinding) -> GenericResult<&mut Self> {
        let read = self.gpu.contexts.read().unwrap();
        let objects = read.get(&self.context)
            .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;

        self.builder.copy_buffer(CopyBufferInfo::buffers(
            objects.get_buffer(src_binding)
                .ok_or_else(||anyhow::anyhow!("Source buffer does not exist"))?,
            objects.get_buffer(dst_binding)
                .ok_or_else(||anyhow::anyhow!("Destination buffer does not exist"))?,
        ))?;

        drop(read);
        Ok(self)
    }

    pub fn update_buffer_with_stored_array(&mut self, binding: ContextBinding, data: &StoredArray,
                                           checksum: BufferChecksumMethod) -> GenericResult<&mut Self> {
        match data {
            StoredArray::Memory { data } => self.update_buffer_with_memory(binding, data, checksum),
            StoredArray::GpuLocal { data, .. } => self.update_buffer_with_buffer(binding, data.clone()),
        }?;
        Ok(self)
    }

    pub fn update_buffer_with_val(&mut self,binding: ContextBinding, value: f32) -> GenericResult<&mut Self> {
        let read = self.gpu.contexts.read().unwrap();
        let objects = read.get(&self.context)
            .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;

        let mut info = FillBufferInfo::dst_buffer(objects.get_buffer(binding)
            .ok_or_else(||anyhow::anyhow!("Destination buffer does not exist"))?);
        info.data = value.to_bits();

        self.builder.fill_buffer(info)?;
        drop(read);
        Ok(self)
    }

    pub fn dispatch(&mut self, shader_name: &str, total_times: [u32; 3], block_size: [u32; 3]) -> GenericResult<&mut Self> {
        let group_counts = Self::create_groups(total_times, block_size)?;

        let read = self.gpu.contexts.read().unwrap();
        let context = read.get(&self.context)
            .ok_or_else(|| anyhow::anyhow!("Id not found in gpu.pipeline_objects"))?;
        let objects = context.get_pipeline_objects(shader_name);
        let pipeline = objects.pipeline.clone();
        let descriptors = objects.descriptor_set.clone();

        self.builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                descriptors,
            )
            .dispatch(group_counts)?;

        drop(read);
        Ok(self)
    }

    pub fn finish(self) -> GenericResult<GpuBuffer> {
        let Self {builder, gpu, context, ..} = self;

        let cmd = builder.build()?;
        gpu.exec_cmd(cmd)?.wait(None)?;

        // Necessary to avoid memory leaks TODO: review
        gpu.cmd_alloc.clear(gpu.queue.queue_family_index());
        gpu.descriptor_alloc.clear_all();

        let read = gpu.contexts.read().unwrap();
        // TODO: improve
        Ok(read[&context].get_buffer(ContextBinding(0)).unwrap())
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
