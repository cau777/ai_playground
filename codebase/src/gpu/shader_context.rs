use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::DeviceLocalBuffer;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::pipeline::{ComputePipeline, Pipeline};
use vulkano::shader::{ShaderCreationError, ShaderModule, SpecializationConstants};
use crate::gpu::buffers::{CpuBuffer, GpuBuffer};
use crate::gpu::gpu_data::GlobalGpu;
use crate::utils::GenericResult;

pub struct ContextSharedBuffer {
    pub aux_buffer: Option<CpuBuffer>,
    pub buffer: GpuBuffer,
    pub length: u64,
    pub checksums: Vec<u64>,
}

pub struct ShaderContext {
    buffers: Vec<ContextSharedBuffer>,
    pipelines: HashMap<String, PipelineObjects>,
}

pub struct PipelineObjects {
    pub pipeline: Arc<ComputePipeline>,
    pub descriptor_set: Arc<PersistentDescriptorSet>,
}

type LoadModuleResult = Result<Arc<ShaderModule>, ShaderCreationError>;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ContextBinding(pub usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ShaderBinding(pub usize);

pub type ShaderContextKey = String;

impl ShaderContext {
    pub fn register(key: &ShaderContextKey, gpu: GlobalGpu, buffers_lengths: &[u64],
                    create_fn: impl FnOnce(ContextBuilder) -> GenericResult<ContextBuilder>) -> GenericResult<()> {
        let should_create = {
            let read = gpu.contexts.read().unwrap();
            let prev = read.get(key);
            prev.is_none() || std::iter::zip(&prev.unwrap().buffers, buffers_lengths.iter())
                .any(|(prev, length)| prev.length != *length)
        };

        if should_create {
            let builder = ContextBuilder::new(buffers_lengths, gpu.clone())?;
            let mut write = gpu.contexts.write().unwrap();
            write.insert(key.to_owned(), (create_fn)(builder)?.finish()?);
        }

        Ok(())
    }

    pub fn print_buffer(&self, index: ContextBinding, gpu: GlobalGpu) {
        let obj = &self.buffers[index.0];
        println!("\n({:?}) -> {:?}\n", obj.length,
                 crate::gpu::buffers::download_array_from_gpu(&obj.buffer, vec![obj.length as usize], &gpu)
                     .unwrap().iter().take(50).collect::<Vec<_>>());
    }

    pub fn get_buffer_object(&self, index: ContextBinding) -> Option<&ContextSharedBuffer> {
        self.buffers.get(index.0)
    }

    pub fn get_buffer_object_mut(&mut self, index: ContextBinding) -> Option<&mut ContextSharedBuffer> {
        self.buffers.get_mut(index.0)
    }

    pub fn get_buffer(&self, index: ContextBinding) -> Option<GpuBuffer> {
        self.buffers.get(index.0).map(|o| o.buffer.clone())
    }

    pub fn get_pipeline_objects(&self, name: &str) -> &PipelineObjects {
        self.pipelines.get(name)
            .ok_or_else(|| anyhow::anyhow!("Pipeline {} was not found", name))
            .unwrap()
    }
}

pub struct ContextBuilder {
    buffers: Vec<ContextSharedBuffer>,
    pipelines: HashMap<String, PipelineObjects>,
    gpu: GlobalGpu,
}

impl ContextBuilder {
    pub fn new(buffer_lengths: &[u64], gpu: GlobalGpu) -> GenericResult<Self> {
        let mut buffers = Vec::new();

        for &length in buffer_lengths {
            let buffer = DeviceLocalBuffer::<[f32]>::array(
                &gpu.memory_alloc,
                length as u64,
                vulkano::buffer::BufferUsage {
                    storage_buffer: true,
                    transfer_dst: true,
                    transfer_src: true,
                    ..vulkano::buffer::BufferUsage::empty()
                },
                gpu.device.active_queue_family_indices().iter().copied(),
            )?;

            buffers.push(ContextSharedBuffer {
                aux_buffer: None,
                length,
                checksums: vec![],
                buffer,
            });
        }

        Ok(Self {
            pipelines: HashMap::new(),
            buffers,
            gpu,
        })
    }

    pub fn register_shader(&mut self, name: &str,
                           load_fn: fn(Arc<Device>) -> LoadModuleResult,
                           bindings: Vec<(ContextBinding, ShaderBinding)>,
                           constants: &impl SpecializationConstants) -> GenericResult<()> {
        let shader = (load_fn)(self.gpu.device.clone())?;
        let entry = shader.entry_point("main")
            .ok_or_else(|| anyhow::anyhow!("No entry named main found"))?; // Currently, different entry points are not supported

        let pipeline = ComputePipeline::new(
            self.gpu.device.clone(),
            entry,
            constants,
            self.gpu.cache.clone(),
            |_| {},
        )?;

        let mut writes = Vec::new();

        for (ctx, bind) in bindings {
            writes.push(WriteDescriptorSet::buffer(
                bind.0 as u32,
                self.buffers.get(ctx.0)
                    .ok_or_else(|| anyhow::anyhow!("Context shader index was not registered"))?
                    .buffer.clone(),
            ));
        }

        let layouts = pipeline.layout().set_layouts();
        let layout = layouts.get(0)
            .ok_or_else(|| anyhow::anyhow!("No layouts found"))?;

        let descriptor_set = PersistentDescriptorSet::new(
            &self.gpu.descriptor_alloc,
            layout.clone(),
            writes,
        )?;

        self.pipelines.insert(name.to_owned(), PipelineObjects {
            pipeline,
            descriptor_set,
        });

        Ok(())
    }

    pub fn finish(self) -> GenericResult<ShaderContext> {
        Ok(ShaderContext {
            buffers: self.buffers,
            pipelines: self.pipelines,
        })
    }
}