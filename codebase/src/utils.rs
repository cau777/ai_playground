use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, ArrayView4, azip, IxDyn};

pub type ArrayF<D> = Array<f32, D>;
pub type Array1F = Array1<f32>;
pub type Array2F = Array2<f32>;
// pub type Array3F = Array3<f32>;
pub type Array4F = Array4<f32>;
pub type ArrayDynF = Array<f32, IxDyn>;

// pub type ArrayView1F<'a> = ArrayView1<'a, f32>;
// pub type ArrayView2F<'a> = ArrayView2<'a, f32>;
// pub type ArrayView3F<'a> = ArrayView3<'a, f32>;
// pub type ArrayView4F<'a> = ArrayView4<'a, f32>;

pub fn arrays_almost_equal<D: ndarray::Dimension>(arr1: &ArrayF<D>, arr2: &ArrayF<D>) -> bool{
    azip!(arr1, arr2).all(|a, b| (a - b).abs() < 0.001)
}
