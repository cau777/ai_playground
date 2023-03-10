use vulkano::buffer::{CpuAccessibleBuffer, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use std::sync::Arc;
use crate::ArrayDynF;
use crate::gpu::gpu_data::GlobalGpu;
use crate::utils::{GenericResult, shape_length};

pub fn upload_array_to_gpu(array: &ArrayDynF, gpu: &GlobalGpu) -> GenericResult<GpuBuffer> {
    let cpu_buffer = CpuAccessibleBuffer::from_iter(
        &gpu.memory_alloc,
        vulkano::buffer::BufferUsage {
            transfer_src: true,
            ..vulkano::buffer::BufferUsage::empty()
        },
        false,
        array.iter().copied(),
    )?;

    let device_local_buffer = DeviceLocalBuffer::<[f32]>::array(
        &gpu.memory_alloc,
        cpu_buffer.len(),
        vulkano::buffer::BufferUsage {
            storage_buffer: true,
            transfer_dst: true,
            transfer_src: true,
            ..vulkano::buffer::BufferUsage::empty()
        },
        gpu.device.active_queue_family_indices().iter().copied(),
    )?;

    let mut builder = AutoCommandBufferBuilder::primary(
        &gpu.cmd_alloc,
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder.copy_buffer(CopyBufferInfo::buffers(
        cpu_buffer,
        device_local_buffer.clone(),
    ))?;

    gpu.exec_cmd(builder.build()?)?.wait(None)?;
    Ok(device_local_buffer)
}


pub fn download_array_from_gpu(buffer: &GpuBuffer, shape: Vec<usize>, gpu: &GlobalGpu) -> GenericResult<ArrayDynF> {
    let shape_len = shape_length(&shape) as u64;
    assert_eq!(buffer.len(), shape_len);

    let cpu_buffer = unsafe {
        CpuAccessibleBuffer::uninitialized_array(
            &gpu.memory_alloc,
            shape_len,
            vulkano::buffer::BufferUsage {
                transfer_dst: true,
                ..vulkano::buffer::BufferUsage::empty()
            },
            true,
        )
    }?;

    let mut builder = AutoCommandBufferBuilder::primary(
        &gpu.cmd_alloc,
        gpu.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder.copy_buffer(CopyBufferInfo::buffers(buffer.clone(), cpu_buffer.clone()))?;


    gpu.exec_cmd(builder.build()?)?.wait(None)?;

    let read = cpu_buffer.read();
    let vec = read?.to_vec();
    Ok(ArrayDynF::from_shape_vec(shape, vec)?)
}

pub type GpuBuffer = Arc<DeviceLocalBuffer<[f32]>>;
