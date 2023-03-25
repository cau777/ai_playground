use ndarray::{Array, Array0, Array1, Array2, Array3, Array4, Array5, azip, Dimension, IxDyn};
// TODO: (repo) add data files sources
pub use ndarray;

pub type GenericResult<T> = anyhow::Result<T>;

type F = f32;
pub type ArrayF<D> = Array<F, D>;
pub type Array0F = Array0<F>;
pub type Array1F = Array1<F>;
pub type Array2F = Array2<F>;
pub type Array3F = Array3<F>;
pub type Array4F = Array4<F>;
pub type Array5F = Array5<F>;
pub type ArrayDynF = Array<F, IxDyn>;

// pub type ArrayView1F<'a> = ArrayView1<'a, f32>;
// pub type ArrayView2F<'a> = ArrayView2<'a, f32>;
// pub type ArrayView3F<'a> = ArrayView3<'a, f32>;
// pub type ArrayView4F<'a> = ArrayView4<'a, f32>;

pub trait ShapeAsArray<const D: usize> {
    fn shape_arr(&self) -> [usize; D];
}

impl<T> ShapeAsArray<2> for Array2<T> {
    fn shape_arr(&self) -> [usize; 2] {
        self.shape().try_into().unwrap()
    }
}

impl<T> ShapeAsArray<3> for Array3<T> {
    fn shape_arr(&self) -> [usize; 3] {
        self.shape().try_into().unwrap()
    }
}

impl<T> ShapeAsArray<4> for Array4<T> {
    fn shape_arr(&self) -> [usize; 4] {
        self.shape().try_into().unwrap()
    }
}

pub trait GetBatchSize {
    fn batch_size(&self) -> usize;
}

impl<T, D: Dimension> GetBatchSize for Array<T, D> {
    #[inline]
    fn batch_size(&self) -> usize {
        self.shape()[0]
    }
}

#[inline]
pub fn arrays_almost_equal<D: Dimension>(arr1: &ArrayF<D>, arr2: &ArrayF<D>) -> bool {
    azip!(arr1, arr2).all(|a, b| (a - b).abs() < 0.001)
}

#[inline]
pub fn lerp_arrays<D: Dimension>(a: &ArrayF<D>, b: &ArrayF<D>, t: F) -> ArrayF<D> {
    a + (b - a) * t
}

#[inline]
pub fn get_dims_after_filter_dyn(shape: &[usize], size: usize, stride: usize) -> Vec<usize> {
    let len = shape.len();
    let mut result = Vec::with_capacity(len);
    for i in 0..(len - 2) {
        result.push(shape[i]);
    }

    result.push((shape[len - 2] - size) / stride + 1);
    result.push((shape[len - 1] - size) / stride + 1);
    result
}

#[inline]
pub fn get_dims_after_filter_4(shape: &[usize], size: usize, stride: usize) -> [usize; 4] {
    [
        shape[0],
        shape[1],
        (shape[2] - size) / stride + 1,
        (shape[3] - size) / stride + 1
    ]
}

pub fn as_array<const N: usize, T: Default + Copy>(slice: &[T]) -> [T; N] {
    if slice.len() != N {
        panic!("Invalid slice length");
    } else {
        let mut result = [T::default(); N];
        result[..N].copy_from_slice(&slice[..N]);
        result
    }
}

pub const EPSILON: f32 = 0.0000001;

#[inline]
pub fn shape_length(shape: &[usize]) -> usize {
    shape.iter().copied()
        .chain(1..=1) // Just append 1 to the iterator
        .reduce(|a, b| a * b).unwrap_or_default()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_get_dims_after_filter() {
        assert_eq!(get_dims_after_filter_dyn(&[3, 3], 1, 1), vec![3, 3]);
        assert_eq!(get_dims_after_filter_dyn(&[1, 4, 4], 1, 1), vec![1, 4, 4]);
        assert_eq!(get_dims_after_filter_dyn(&[1, 1, 4, 4], 1, 1), vec![1, 1, 4, 4]);

        assert_eq!(get_dims_after_filter_dyn(&[1, 1, 4, 4], 3, 1), vec![1, 1, 2, 2]);
        assert_eq!(get_dims_after_filter_dyn(&[1, 1, 4, 4], 2, 1), vec![1, 1, 3, 3]);

        assert_eq!(get_dims_after_filter_dyn(&[1, 1, 6, 6], 2, 3), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_len() {
        assert_eq!(shape_length(&[]), Array0F::zeros(()).len());
        assert_eq!(shape_length(&[10]), 10);
        assert_eq!(shape_length(&[5, 5]), 25);
        assert_eq!(shape_length(&[3, 3, 3]), 27);
    }
}

