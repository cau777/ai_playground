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
    pub element_size: u64,
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
    pub fn register(key: &ShaderContextKey, gpu: GlobalGpu, buffer_configs: &[BufferConfig],
                    create_fn: impl FnOnce(ContextBuilder) -> GenericResult<ContextBuilder>) -> GenericResult<()> {
        let should_create = {
            let read = gpu.contexts.read().unwrap();
            let prev = read.get(key);
            prev.is_none() || std::iter::zip(&prev.unwrap().buffers, buffer_configs.iter())
                .any(|(prev, new)| {
                    prev.length != new.length || prev.element_size != new.ty.size()
                })
        };

        if should_create {
            let builder = ContextBuilder::new(buffer_configs, gpu.clone())?;
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

#[derive(Copy, Clone)]
pub struct BufferConfig {
    length: u64,
    ty: BufferType,
}

#[derive(Copy, Clone)]
pub enum BufferType {
    Float,
    Bool,
    Uint,
}

impl BufferType {
    pub fn size(&self) -> u64 {
        match self {
            BufferType::Bool => 1,
            BufferType::Float => 4,
            BufferType::Uint => 4,
        }
    }
}

impl BufferConfig {
    pub fn floats(length: impl TryInto<u64>) -> Self {
        Self {
            length: length.try_into().map_err(|_| anyhow::anyhow!("Buffer size too big")).unwrap(),
            ty: BufferType::Float,
        }
    }
    pub fn bools(length: impl TryInto<u64>) -> Self {
        Self {
            length: length.try_into().map_err(|_| anyhow::anyhow!("Buffer size too big")).unwrap(),
            ty: BufferType::Bool,
        }
    }
    pub fn uints(length: impl TryInto<u64>) -> Self {
        Self {
            length: length.try_into().map_err(|_| anyhow::anyhow!("Buffer size too big")).unwrap(),
            ty: BufferType::Uint,
        }
    }
}

impl ContextBuilder {
    pub fn new(buffer_configs: &[BufferConfig], gpu: GlobalGpu) -> GenericResult<Self> {
        let mut buffers = Vec::new();

        for &BufferConfig { length, ty } in buffer_configs {
            let buffer = match ty {
                BufferType::Bool => Self::create_buffer::<u8>(length, &gpu),
                BufferType::Float => Self::create_buffer::<f32>(length, &gpu),
                BufferType::Uint => Self::create_buffer::<u32>(length, &gpu),
            }?;

            buffers.push(ContextSharedBuffer {
                aux_buffer: None,
                length,
                checksums: vec![],
                buffer,
                element_size: ty.size(),
            });
        }

        Ok(Self {
            pipelines: HashMap::new(),
            buffers,
            gpu,
        })
    }

    fn create_buffer<T>(length: u64, gpu: &GlobalGpu) -> GenericResult<GpuBuffer>
        where [T]: vulkano::buffer::BufferContents {
        Ok(DeviceLocalBuffer::<[T]>::array(
            &gpu.memory_alloc,
            length,
            vulkano::buffer::BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                transfer_src: true,
                ..vulkano::buffer::BufferUsage::empty()
            },
            gpu.device.active_queue_family_indices().iter().copied(),
        )?)
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