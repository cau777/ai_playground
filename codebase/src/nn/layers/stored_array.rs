use std::sync::Arc;
use vulkano::buffer::{CpuAccessibleBuffer, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use crate::ArrayDynF;
use crate::gpu::gpu_data::GlobalGpu;
use crate::nn::utils::shape_length;
use crate::utils::GenericResult;

pub type GpuBuffer = Arc<DeviceLocalBuffer<[f32]>>;

pub enum StoredArray {
    Memory { data: ArrayDynF },
    GpuLocal { gpu: GlobalGpu, data: GpuBuffer, shape: Vec<usize> },
}

impl From<ArrayDynF> for StoredArray {
    fn from(array: ArrayDynF) -> Self {
        StoredArray::Memory { data: array }
    }
}

impl StoredArray {
    pub fn into_memory(self) -> GenericResult<ArrayDynF> {
        match self {
            StoredArray::Memory { data } => Ok(data),
            StoredArray::GpuLocal { data, gpu, shape } => {
                let cpu_buffer = unsafe {
                    CpuAccessibleBuffer::uninitialized_array(
                        &gpu.memory_alloc,
                        data.len(),
                        vulkano::buffer::BufferUsage {
                            transfer_dst: true,
                            ..vulkano::buffer::BufferUsage::empty()
                        },
                        true,
                    )
                }?;
                // let debug: Vec<_> = (0..data.len()).into_iter().map(|o| o as f32).collect();
                // let cpu_buffer =
                //     CpuAccessibleBuffer::from_iter(
                //         &gpu.memory_alloc,
                //         vulkano::buffer::BufferUsage {
                //             transfer_dst: true,
                //             ..vulkano::buffer::BufferUsage::empty()
                //         },
                //         true,
                //         debug.into_iter(),
                //     )?;

                let mut builder = AutoCommandBufferBuilder::primary(
                    &gpu.cmd_alloc,
                    gpu.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )?;
                // builder.copy_buffer(CopyBufferInfo::buffers(data, cpu_buffer.clone()))?;
                // let mut fill = FillBufferInfo::dst_buffer(data.clone());
                // fill.data = 9;
                // builder.fill_buffer(fill)?;
                builder.copy_buffer(CopyBufferInfo::buffers(data, cpu_buffer.clone()))?;


                gpu.exec_cmd(builder.build()?)?.wait(None)?;

                let read = cpu_buffer.read();
                let vec = read?.to_vec();
                Ok(ArrayDynF::from_shape_vec(shape, vec)?)
            }
        }
    }

    pub fn into_gpu_local(self, gpu: GlobalGpu) -> GenericResult<GpuBuffer> {
        match self {
            StoredArray::Memory { data } => {
                let cpu_buffer = CpuAccessibleBuffer::from_iter(
                    &gpu.memory_alloc,
                    vulkano::buffer::BufferUsage {
                        transfer_src: true,
                        ..vulkano::buffer::BufferUsage::empty()
                    },
                    false,
                    data.iter().copied(),
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
            StoredArray::GpuLocal { data, .. } => { Ok(data) }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            StoredArray::Memory { data } => data.shape(),
            StoredArray::GpuLocal { shape, .. } => &shape,
        }
    }

    pub fn len(&self) -> usize {
        shape_length(self.shape())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array0;
    use crate::gpu::gpu_data::GpuData;
    use crate::utils::{Array0F, Array1F, Array2F, Array3F, arrays_almost_equal};
    use super::*;

    #[test]
    fn test_cpu_gpu_cpu() {
        let mut i = 0;
        let initial_array = Array3F::from_shape_simple_fn((3, 4, 5), || {
            i += 1;
            i as f32
        }).into_dyn();
        let shape = initial_array.shape().to_vec();

        let stored = StoredArray::Memory { data: initial_array.clone() };
        let gpu = GpuData::new_global().unwrap();
        let buffer = stored.into_gpu_local(gpu.clone()).unwrap();

        let stored = StoredArray::GpuLocal { gpu, shape, data: buffer };
        let final_array = stored.into_memory().unwrap();

        assert!(arrays_almost_equal(&initial_array, &final_array));
    }
}