#version 450
layout(local_size_x = 8, local_size_y = 2, local_size_z = 2) in;
layout(set = 0, binding = 0) buffer ResultData {
    float data[];
} result_data;

layout(set = 0, binding = 1) buffer KernelData {
    float data[];
} kernel_data;

layout(set = 0, binding = 2) buffer GradData {
    float data[];
} grad_data;

layout(constant_id = 0) const uint batch_size = 0;
layout(constant_id = 1) const uint in_channels = 0;
layout(constant_id = 2) const uint out_channels = 0;
layout(constant_id = 3) const uint kernel_size = 0;
layout(constant_id = 4) const uint stride = 0;
layout(constant_id = 5) const uint padding = 0;

layout(constant_id = 6) const uint input_width = 0;
layout(constant_id = 7) const uint input_height = 0;

layout(constant_id = 8) const uint grad_width = 0;
layout(constant_id = 9) const uint grad_height = 0;

layout(constant_id = 10) const uint out_width = 0;
layout(constant_id = 11) const uint out_height = 0;

void main() {
    const uint b_and_in_c = gl_GlobalInvocationID.x;
    const uint b = b_and_in_c / in_channels;
    const uint in_c = b_and_in_c % in_channels;
    const uint h = gl_GlobalInvocationID.y;
    const uint w = gl_GlobalInvocationID.z;

    const uint padded_w = w + padding;
    const uint padded_h = h + padding;

    const uint max_h = min(kernel_size, padded_h + 1);// Asserts the condition grad_h >= 0
    const uint min_h = max(padded_h / stride + 1, grad_height) - grad_height;// Asserts the condition grad_h < grad_max_h

    float result = 0.0;

    // Rest of the division by the stride to get the position relative to the filter
    uint kernel_h = max(padded_h % stride, min_h);

    uint grad_sections[3];
    grad_sections[0] = out_channels * grad_height * grad_width;
    grad_sections[1] = grad_height * grad_width;
    grad_sections[2] = grad_width;

    uint kernel_sections[3];
    kernel_sections[0] = in_channels*kernel_size*kernel_size;
    kernel_sections[1] = kernel_size*kernel_size;
    kernel_sections[2] = kernel_size;

    while (kernel_h < max_h) {
        uint grad_h = (padded_h - kernel_h) / stride;
        const uint max_w = min(kernel_size, padded_w + 1);// Asserts the condition grad_w >= 0
        const uint min_w = max(padded_w / stride + 1, grad_width) - grad_width;// Asserts the condition grad_w < grad_max_w

        uint kernel_w = max(padded_w % stride, min_w);
        while (kernel_w < max_w) {
            const uint grad_w = (padded_w - kernel_w) / stride;

            for (uint out_c = 0; out_c < out_channels; out_c++) {
                const float g = grad_data.data[b*grad_sections[0] + out_c*grad_sections[1] + grad_h*grad_sections[2] + grad_w];
                const float k = kernel_data.data[out_c*kernel_sections[0] + in_c*kernel_sections[1] + kernel_h*kernel_sections[2] + kernel_w];
                result += g * k;
            }

            kernel_w += stride;
        }

        kernel_h += stride;
    }

    const uint result_index = b*in_channels*out_height*out_width + in_c*out_height*out_width + h*out_width + w;
    result_data.data[result_index] = result;
}

/*
Shader equivalent in Rust
Array4F::from_shape_fn(out_shape, |(b, in_c, h, w)| {
    let padded_h = h + padding;
    let padded_w = w + padding;
    let mut result = 0.0;

    let max_h = kernel_size.min(padded_h + 1);// Asserts the condition grad_h >= 0
    let min_h = (padded_h / stride + 1).max(grad_max_h) - grad_max_h;// Asserts the condition grad_h < grad_max_h

    // Rest of the division by the stride to get the position relative to the filter
    let mut kernel_h = (padded_h % stride).max(min_h);
    while kernel_h < max_h {
        let grad_h = (padded_h - kernel_h) / stride;

        let max_w = kernel_size.min(padded_w + 1); // Asserts the condition grad_w >= 0
        let min_w = (padded_w / stride + 1).max(grad_max_w) - grad_max_w; // Asserts the condition grad_w < grad_max_w

        let mut kernel_w = (padded_w % stride).max(min_w);
        while kernel_w < max_w {
            let grad_w = (padded_w - kernel_w) / stride;

            for out_c in 0..*out_channels {
                result += grad[(b, out_c, grad_h, grad_w)] * kernel[(out_c, in_c, kernel_h, kernel_w)]
            }

            kernel_w += stride;
        }
        kernel_h += stride;
    }
    result
})
 */