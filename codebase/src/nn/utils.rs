use ndarray::s;
use crate::Array4F;

pub fn pad4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2];
    let width = shape[3];
    let mut result = Array4F::zeros(
        (
            shape[0],
            shape[1],
            height + 2 * padding,
            width + 2 * padding,
        ),
    );
    let mut slice = result.slice_mut(s![
        ..,
        ..,
        padding..height + padding,
        padding..width + padding
    ]);
    slice.assign(&array);
    result
}

pub fn remove_padding_4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2] - padding;
    let width = shape[3] - padding;
    array.slice_move(s![.., .., padding..height, padding..width])
}