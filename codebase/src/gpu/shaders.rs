pub mod convolution_inputs_grad {
    pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/convolution_inputs_grad.glsl"
    }
}

pub mod convolution_forward {
    pub mod forward {
        pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

        vulkano_shaders::shader! {
            ty: "compute",
            path: "./src/gpu/shaders/convolution_forward/forward.glsl"
        }
    }

    pub mod validate_cache_1 {
        pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

        vulkano_shaders::shader! {
            ty: "compute",
            path: "./src/gpu/shaders/convolution_forward/validate_cache_1.glsl"
        }
    }

    pub mod validate_cache_2 {
        pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

        vulkano_shaders::shader! {
            ty: "compute",
            path: "./src/gpu/shaders/convolution_forward/validate_cache_2.glsl"
        }
    }
}

pub mod relu_forward {
    pub const BLOCK_SIZE: [u32; 3] = [8, 1, 1];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/relu_forward.glsl"
    }
}

pub mod max_pool_forward {
    pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/max_pool_forward.glsl"
    }
}

pub mod dense_forward {
    pub const BLOCK_SIZE: [u32; 3] = [8, 4, 1];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/dense_forward.glsl"
    }
}