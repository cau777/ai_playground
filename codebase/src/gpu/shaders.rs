pub mod convolution_inputs_grad {
    pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/convolution_inputs_grad.glsl"
    }
}

pub mod convolution_forward {
    pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/convolution_forward.glsl"
    }
}