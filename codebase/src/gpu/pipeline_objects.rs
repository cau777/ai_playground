use std::collections::HashMap;
use std::sync::Arc;
use crate::gpu::buffers::GpuBuffer;

pub struct PipelineObjects {
    pub pipelines: HashMap<String, Arc<vulkano::pipeline::ComputePipeline>>,
    pub buffers: Vec<BufferObject>,
    pub descriptor_set: Arc<vulkano::descriptor_set::PersistentDescriptorSet>,
}

pub struct BufferObject {
    pub buffer: GpuBuffer,
    pub length: usize,
    pub checksums: Vec<u64>,
}
