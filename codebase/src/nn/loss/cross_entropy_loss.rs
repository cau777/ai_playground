use std::iter::zip;

use crate::nn::loss::loss_func::LossFuncOps;
use crate::utils::{Array1F, Array2F, ArrayDynF};

pub struct CrossEntropyLoss {}

fn softmax(array: Array2F) -> Array2F {
    let max = array
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare NaN values"))
        .unwrap_or(0.0);
    let mut e = (array - max).mapv_into(f32::exp);

    // In-place operation that divide each value of a batch by the sum of that batch
    e.outer_iter_mut().for_each(|mut batch| {
        let sum: f32 = batch.iter().sum();
        batch.iter_mut().for_each(|o| *o /= sum);
    });
    e
}

impl LossFuncOps for CrossEntropyLoss {
    fn calc_loss(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        let expected: Array2F = expected.clone().into_dimensionality().unwrap();
        let actual: Array2F = actual.clone().into_dimensionality().unwrap();

        let prob = softmax(actual);
        let iter = zip(expected.outer_iter(), prob.outer_iter()).map(|(expected, actual)| {
            let label = expected
                .iter()
                .enumerate()
                .reduce(|acc, val| if val.1 > acc.1 { val } else { acc })
                .map(|o| o.0)
                .unwrap_or(0); // Get the index of the highest value
            -actual[label].ln()
        });
        Array1F::from_iter(iter).into_dyn()
    }

    fn calc_loss_grad(expected: &ArrayDynF, actual: &ArrayDynF) -> ArrayDynF {
        let actual: Array2F = actual.clone().into_dimensionality().unwrap();
        let mut prob = softmax(actual);
        prob -= expected;
        (-prob).into_dyn()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::arrays_almost_equal;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_softmax() {
        let inputs = get_inputs_actual();
        let expected: Array2F = array![
            [0.34200877, 0.37797814, 0.28001309],
            [0.2693075, 0.401_759_6, 0.328_932_9],
            [0.26030255, 0.35137169, 0.38832577],
            [0.37797814, 0.34200877, 0.28001309]
        ];
        let result = softmax(inputs);
        assert!(arrays_almost_equal(&result, &expected));
    }

    #[test]
    fn test_calc_loss() {
        let inputs_actual = get_inputs_actual();
        let inputs_expected = get_inputs_expected();

        let expected: ArrayDynF = array![1.072_918_7, 1.111_901_2, 1.3459103, 0.972_918_6].into_dyn();
        let result =
            CrossEntropyLoss::calc_loss(&inputs_expected.into_dyn(), &inputs_actual.into_dyn());
        assert!(arrays_almost_equal(&result, &expected));
    }

    #[test]
    fn test_calc_loss_grad() {
        let inputs_actual = get_inputs_actual();
        let inputs_expected = get_inputs_expected();
        let expected = array![
            [0.25799123, -0.27797814, -0.28001309],
            [0.1306925, 0.09824042, 0.47106708],
            [0.03969745, -0.25137169, -0.28832577],
            [0.522_021_9, 0.15799123, 0.11998691]
        ].into_dyn();
        
        let result = CrossEntropyLoss::calc_loss_grad(
            &inputs_expected.into_dyn(),
            &inputs_actual.into_dyn(),
        );
        assert!(arrays_almost_equal(&result, &expected));
    }

    fn get_inputs_actual() -> Array2F {
        array![
            [0.6, 0.7, 0.4],
            [0.1, 0.5, 0.3],
            [0.2, 0.5, 0.6],
            [0.7, 0.6, 0.4]
        ]
    }

    fn get_inputs_expected() -> Array2F {
        array![
            [0.6, 0.1, 0.],
            [0.4, 0.5, 0.8],
            [0.3, 0.1, 0.1],
            [0.9, 0.5, 0.4]
        ]
    }
}
