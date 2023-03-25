use crate::ArrayDynF;
use crate::gpu::buffers::{download_array_from_gpu, GpuBuffer, upload_array_to_gpu};
use crate::gpu::gpu_data::GlobalGpu;
use crate::utils::shape_length;
use crate::utils::GenericResult;

#[derive(Clone)]
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
    pub fn to_memory(&self) -> GenericResult<ArrayDynF> {
        match self {
            StoredArray::Memory { data } => Ok(data.clone()),
            StoredArray::GpuLocal { data, gpu, shape } => download_array_from_gpu(&data, shape.clone(), &gpu),
        }
    }

    pub fn into_memory(self) -> GenericResult<ArrayDynF> {
        match self {
            StoredArray::Memory { data } => Ok(data),
            StoredArray::GpuLocal { data, gpu, shape } => download_array_from_gpu(&data, shape, &gpu),
        }
    }

    pub fn into_gpu_local(self, gpu: GlobalGpu) -> GenericResult<GpuBuffer> {
        match self {
            StoredArray::Memory { data } => upload_array_to_gpu(&data, &gpu),
            StoredArray::GpuLocal { data, .. } => { Ok(data) }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            StoredArray::Memory { data } => data.shape(),
            StoredArray::GpuLocal { shape, .. } => shape,
        }
    }

    pub fn len(&self) -> usize {
        shape_length(self.shape())
    }
}

#[cfg(test)]
mod tests {
    use crate::gpu::gpu_data::GpuData;
    use crate::utils::{Array3F, arrays_almost_equal};
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