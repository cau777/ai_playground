use ndarray::{array, stack, Axis};
use crate::{Array4F, ArrayDynF};
use crate::nn::layers::filtering::convolution::ConvolutionConfig;
use crate::nn::layers::filtering::convolution::ConvolutionInitMode::Kernel;
use crate::nn::layers::nn_layers::GenericStorage;
use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
use crate::nn::lr_calculators::lr_calculator::LrCalc;

pub fn get_inputs() -> ArrayDynF {
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
    stack![Axis(0), inputs].into_dyn()
}

pub fn get_kernels() -> Array4F {
    stack![
            Axis(0),
            array![
                [[-0.2341, -0.41141], [-0.03269, -0.35668]],
                [[0.45318, 0.38312], [0.41303, -0.66184]]
            ],
            array![
                [[-0.87622, 0.50122], [0.2724, 0.94758]],
                [[-0.38468, -0.70155], [-0.31623, -0.27944]]
            ],
            array![
                [[-0.61662, -0.21975], [0.45739, 0.13252]],
                [[-0.69169, 0.34276], [0.22805, -0.23069]]
            ]
        ]
}

pub fn get_forward_result() -> ArrayDynF {
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
    stack![Axis(0), result].into_dyn()
}

pub fn get_backward_result() -> ArrayDynF {
    let result = array![
            [
                [-0.4668, -0.34546, -0.96733, -0.59356, -1.39406],
                [0.00082, -0.29748, -0.39212, -0.79371, -0.18983],
                [0.24772, 0.07721, -0.28081, 0.19632, 0.15916],
                [-0.22853, -1.17987, -0.2272, -2.06669, -0.01049],
                [0.17371, 0.62756, 0.38954, 0.98389, 1.06211]
            ],
            [
                [-0.00396, 0.59767, -0.62592, 0.52924, -0.0602],
                [-0.1467, 0.19314, 0.40909, 0.11018, 0.0849],
                [-0.61417, 0.38781, -0.55568, 0.28683, -0.83175],
                [0.10623, -0.75989, 0.27379, -1.21815, -0.0459],
                [-0.97521, 0.31429, -0.67425, 0.23328, -1.05826]
            ]
        ];
    stack![Axis(0), result].into_dyn()
}

pub fn get_grad() -> ArrayDynF {
    get_forward_result() * -2.0
}

pub fn get_kernels_grad() -> ArrayDynF {
    let grad: Array4F = stack![
            Axis(0),
            array![
                [[0.046632495, 0.04863675],
                 [0.07071798, 0.08882607]],
                [[0.03103437, 0.04097781],
                 [0.07444477, 0.12355672]]
            ],
            array![
                [[0.03151616, 0.035256498],
                 [-0.0008683999, 0.008833285]],
                [[0.03629486, 0.026376387],
                 [0.03135302, 0.039738197]]
            ],
            array![
                [[0.045678787, 0.059112985],
                 [0.028015418, 0.025232412]],
                [[0.063403, 0.021337919],
                 [0.048176333, 0.03950825]]
            ]
        ];
    grad.into_dyn()
}

pub fn get_config() -> ConvolutionConfig {
    ConvolutionConfig {
        kernel_size: 2,
        stride: 2,
        padding: 1,
        init_mode: Kernel(get_kernels()),
        lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
        in_channels: 2,
        out_channels: 3,
        cache: false,
    }
}

pub fn get_storage() -> GenericStorage {
    let mut result = GenericStorage::new();
    result.insert("convolution_2_3_2_2_1_0".to_owned(), vec![get_kernels().into_dyn()]);
    result
}