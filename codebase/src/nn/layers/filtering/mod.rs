use ndarray::{Array3, Array4, s, Zip};
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use crate::Array4F;

pub mod convolution;
pub mod max_pool;

fn pad4d(array: Array4F, padding: usize) -> Array4F {
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

fn remove_padding_4d(array: Array4F, padding: usize) -> Array4F {
    let shape = array.shape();
    let height = shape[2] - padding;
    let width = shape[3] - padding;
    array.slice_move(s![.., .., padding..height, padding..width])
}

fn find_useful_from_prev(prev_inputs: &Array4F, prev_result: &Array4F, curr_input: &Array4F, size: usize, stride: usize) -> Array4<Option<f32>> {
    if prev_inputs.shape() != curr_input.shape() {
        panic!("Invalid shapes {:?} {:?}", prev_inputs.shape(), curr_input.shape());
    }

    let [batch, ich, ih, iw]: [usize; 4] = curr_input.shape().try_into().unwrap();
    let [_, och, oh, ow]: [usize; 4] = prev_result.shape().try_into().unwrap();

    // Creates a new array that shows if the each prev_input pixel is different from the current_input_pixel
    let mut channels_different = Array3::from_elem((batch, ih, iw), false);
    Zip::from(prev_inputs.outer_iter())
        .and(curr_input.outer_iter())
        .and(channels_different.outer_iter_mut())
        .into_par_iter()
        .for_each(|(prev_inputs, curr_input, mut result)| {
            for h in 0..ih {
                for w in 0..iw {
                    let mut different = false;
                    let mut c = 0;
                    while c < ich && !different {
                        different |= (prev_inputs[(c, h, w)] - curr_input[(c, h, w)]).abs() > 0.0001;
                        c += 1;
                    }
                    result[(h, w)] = different;
                }
            }
        });

    let mut result = Array4::from_elem((batch, och, oh, ow), None);
    Zip::from(channels_different.outer_iter())
        .and(prev_result.outer_iter())
        .and(result.outer_iter_mut())
        .into_par_iter()
        .for_each(|(channels_different, prev_result, mut result)| {
            // This uses the moving window pattern, adapted to be 2D
            // Instead of having to check all element every time the filter is applied, it keeps a count of how many
            // element differ at a given time. When the filter moves, it removes the elements that went out and add the ones that went in.
            // If the count is 0, than every element is equal in that instance
            for h in 0..oh {
                let h_offset = h * stride;
                let mut diff_count = 0;

                for kh in 0..size {
                    for kw in 0..size {
                        diff_count += channels_different[(h_offset + kh, kw)] as u8 as u32;
                    }
                }

                for w in 0..ow {
                    let w_offset = w * stride;
                    if diff_count == 0 {
                        // If the inputs in the in_channel are equal, the output in the out_channel will also be equal
                        // (provided that the kernel is the same)
                        for o in 0..och {
                            result[(o, h, w)] = Some(prev_result[(o, h, w)]);
                        }
                    }

                    // Add elements that went in
                    if w < ow - 1 {
                        for temp_h in 0..size {
                            diff_count += channels_different[(h_offset + temp_h, w_offset + size)] as u8 as u32;
                        }
                    }

                    // Subtract elements that went out of scope
                    for temp_h in 0..size {
                        diff_count -= channels_different[(h_offset + temp_h, w_offset)] as u8 as u32;
                    }
                }
            }
        });
    result
}

#[cfg(test)]
mod tests {
    use ndarray::{array, stack, Axis};
    use super::*;

    #[test]
    fn test_find_useful_from_prev_all_useful() {
        let prev_inputs = get_inputs_1();
        let inputs = get_inputs_1();

        let result = find_useful_from_prev(&prev_inputs, &get_result_1(), &inputs, 3, 1);
        assert!(result.iter().all(|o| o.is_some()))
    }

    #[test]
    fn test_find_useful_from_prev_none_useful() {
        let prev_inputs = get_inputs_1();
        let inputs = get_inputs_1().mapv(|o|-o);

        let result = find_useful_from_prev(&prev_inputs, &get_result_1(), &inputs, 3, 1);
        assert!(result.iter().all(|o| o.is_none()))
    }

    #[test]
    fn test_find_useful_from_prev_some_useful() {
        let prev_inputs = get_inputs_1();
        let mut inputs = get_inputs_1();
        let mut expected = Array4::from_elem((1, 3, 3, 3), true);
        inputs[(0, 1, 1, 1)] = 0.0;
        expected[(0, 0, 0, 0)] = false;
        expected[(0, 1, 0, 0)] = false;
        expected[(0, 2, 0, 0)] = false;

        expected[(0, 0, 1, 0)] = false;
        expected[(0, 1, 1, 0)] = false;
        expected[(0, 2, 1, 0)] = false;

        expected[(0, 0, 0, 1)] = false;
        expected[(0, 1, 0, 1)] = false;
        expected[(0, 2, 0, 1)] = false;

        expected[(0, 0, 1, 1)] = false;
        expected[(0, 1, 1, 1)] = false;
        expected[(0, 2, 1, 1)] = false;

        let result = find_useful_from_prev(&prev_inputs, &get_result_1(), &inputs, 3, 1);
        assert_eq!(result.mapv(|o| o.is_some()), expected)
    }

    //noinspection ALL
    fn get_inputs_1() -> Array4F {
        let inputs = array![
            [
                [0.22537, 0.51686, 0.5185, 0.60037, 0.53262],
                [0.01331, 0.5241, 0.89588, 0.7699, 0.12285],
                [0.29587, 0.61202, 0.72614, 0.4635, 0.76911],
                [0.19163, 0.55787, 0.55078, 0.47223, 0.79188],
                [0.11525, 0.6813, 0.36233, 0.34421, 0.44952]
            ],
            [
                [0.02694, 0.41525, 0.92223, 0.09121, 0.31512],
                [0.52802, 0.32806, 0.44892, 0.01633, 0.09703],
                [0.69259, 0.83594, 0.42432, 0.84877, 0.54679],
                [0.3541, 0.72725, 0.09385, 0.89286, 0.33626],
                [0.89183, 0.29685, 0.30165, 0.80624, 0.83761]
            ]
        ];
        stack![Axis(0), inputs]
    }

    //noinspection ALL
    fn get_result_1() -> Array4F {
        let result = array![
            [
                [-0.09822, -0.6407, -0.38049],
                [-0.3671, -0.38519, -0.48701],
                [-0.57453, -0.22021, -0.29585]
            ],
            [
                [0.20603, 0.24309, 0.55135],
                [-0.27693, 0.02055, -0.25353],
                [-0.29237, -0.20759, -0.56553]
            ],
            [
                [0.02365, 0.18707, 0.2933],
                [0.0575, -0.12417, -0.09843],
                [-0.1112, -0.57813, -0.75988]
            ]
        ];
        stack![Axis(0), result]
    }
}