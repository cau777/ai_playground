use vulkano::buffer::{CpuBufferPool, BufferUsage};
use std::sync::Arc;
use vulkano::memory::allocator::{MemoryUsage, StandardMemoryAllocator};

pub struct Pools {
    pub gpu_only_pool: CpuBufferPool<f32>,
    pub download_pool: CpuBufferPool<f32>,
    pub upload_pool: CpuBufferPool<f32>,
}

impl Pools {
    pub fn new(memory_alloc: Arc<StandardMemoryAllocator>) -> Self {
        Self {
            gpu_only_pool: CpuBufferPool::new(memory_alloc.clone(), BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                ..BufferUsage::empty()
            }, MemoryUsage::Download),
            download_pool: CpuBufferPool::new(memory_alloc.clone(), BufferUsage {
                //storage_buffer: true,
                transfer_dst: true,
                ..BufferUsage::empty()
            }, MemoryUsage::Download),
            upload_pool: CpuBufferPool::new(memory_alloc, BufferUsage {
                //storage_buffer: true,
                transfer_src: true,
                ..BufferUsage::empty()
            }, MemoryUsage::Upload),
        }
    }
}
