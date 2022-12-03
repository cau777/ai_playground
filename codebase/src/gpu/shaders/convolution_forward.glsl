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
} input_data;

layout(constant_id = 0) const uint in_channels = 0;
layout(constant_id = 1) const uint out_channels = 0;
layout(constant_id = 2) const uint kernel_size = 0;
layout(constant_id = 3) const uint stride = 0;

layout(constant_id = 4) const uint input_width = 0;
layout(constant_id = 5) const uint input_height = 0;

layout(constant_id = 6) const uint out_width = 0;
layout(constant_id = 7) const uint out_height = 0;

void main() {
    const uint b_and_in_c = gl_GlobalInvocationID.x;
    const uint b = b_and_in_c / out_channels;
    const uint out_c = b_and_in_c % out_channels;
    const uint h = gl_GlobalInvocationID.y;
    const uint w = gl_GlobalInvocationID.z;

    const uint h_offset = h * stride;
    const uint w_offset = w * stride;

    float result = 0.0;

    uint inputs_section[3];
    inputs_section[0] = in_channels * input_height * input_width;
    inputs_section[1] = input_height * input_width;
    inputs_section[2] = input_width;

    uint kernel_sections[3];
    kernel_sections[0] = in_channels*kernel_size*kernel_size;
    kernel_sections[1] = kernel_size*kernel_size;
    kernel_sections[2] = kernel_size;

    for (uint kh = 0; kh < kernel_size; kh++){
        for (uint kw = 0; kw < kernel_size; kw++) {
            for (uint in_c = 0; in_c < in_channels; in_c++) {
                const float i = input_data.data[b*inputs_section[0] + in_c*inputs_section[1] + (h_offset + kh) * inputs_section[2] + (w_offset + kw)];
                const float k = kernel_data.data[out_c*kernel_sections[0] + in_c*kernel_sections[1] + kh*kernel_sections[2] + kw];
                result += i * k;
            }
        }
    }

    const uint index = b*out_channels*out_height*out_width + out_c*out_height*out_width + h*out_width + w;
    result_data.data[index] = result;
}

/*
Shader equivalent in Rust
Array4F::from_shape_fn((batch_size, layer_config.out_channels, new_height, new_width), |(b, out_c, h, w)| {
    let h_offset = h * stride;
    let w_offset = w * stride;
    let mut result = 0.0;
    for kh in 0..*kernel_size {
        for kw in 0..*kernel_size {
            for in_c in 0..layer_config.in_channels {
                result += inputs[(b, in_c, h_offset + kh, w_offset + kw)] * kernel[(out_c, in_c, kh, kw)];
            }
        }
    }
    result
})
 */