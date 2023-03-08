pub mod convolution_inputs_grad {
    pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/convolution_inputs_grad.glsl"
    }
}

pub mod convolution_forward {
    #[cfg(test)]
    pub const BLOCK_SIZE: [u32; 3] = [1, 2, 2];

    #[cfg(not(test))]
    pub const BLOCK_SIZE: [u32; 3] = [8, 2, 2];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/convolution_forward.glsl"
    }
}

pub mod relu_forward {
    pub const BLOCK_SIZE: [u32; 3] = [8, 1, 1];

    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/gpu/shaders/relu_forward.glsl"
    }
}