#version 450

layout(local_size_x = 8, local_size_y = 2, local_size_z = 2) in;
layout(set = 0, binding = 0) buffer ResultData {
    float data[];
} result_data;

layout(set = 0, binding = 1) buffer KernelData {
    float data[];
} kernel_data;

layout(set = 0, binding = 2) buffer InputData {
    float data[];
} input_data;

layout(set = 0, binding = 3) buffer CacheInvalidData {
    float data[];
} cache_invalid;

layout(constant_id = 0) const uint in_channels = 0;
layout(constant_id = 1) const uint out_channels = 0;
layout(constant_id = 2) const uint kernel_size = 0;
layout(constant_id = 3) const uint stride = 0;

layout(constant_id = 4) const uint input_width = 0;
layout(constant_id = 5) const uint input_height = 0;

layout(constant_id = 6) const uint out_width = 0;
layout(constant_id = 7) const uint out_height = 0;

layout(constant_id = 8) const uint padding = 0;

void main() {
    const uint b_and_in_c = gl_GlobalInvocationID.x;
    const uint b = b_and_in_c / out_channels;
    const uint out_c = b_and_in_c % out_channels;
    const uint h = gl_GlobalInvocationID.y;
    const uint w = gl_GlobalInvocationID.z;

    const uint h_offset = h * stride;
    const uint w_offset = w * stride;

    if (floatBitsToUint(cache_invalid.data[b*out_height*out_width + h*out_width + w]) != 0x1) {
        return;
    }

    float result = 0.0;

    const uint inputs_section_0 = in_channels * input_height * input_width;
    const uint inputs_section_1 = input_height * input_width;
    const uint inputs_section_2 = input_width;

    const uint kernel_sections_0 = in_channels*kernel_size*kernel_size;
    const uint kernel_sections_1 = kernel_size*kernel_size;
    const uint kernel_sections_2 = kernel_size;

    for (uint kh = 0; kh < kernel_size; kh++) {
        for (uint kw = 0; kw < kernel_size; kw++) {
            for (uint in_c = 0; in_c < in_channels; in_c++) {
                const uint input_h = h_offset + kh - padding;
                const uint input_w = w_offset + kw - padding;
                const float i = input_data.data[b*inputs_section_0 + in_c*inputs_section_1 + input_h*inputs_section_2 + input_w];
                const float k = kernel_data.data[out_c*kernel_sections_0 + in_c*kernel_sections_1 + kh*kernel_sections_2 + kw];
                result += i * k * float(input_h < input_height) * float(input_w < input_width);
            }
        }
    }

    const uint result_index = b*out_channels*out_height*out_width + out_c*out_height*out_width + h*out_width + w;
    result_data.data[result_index] = result;
}

/*
Function equivalent in Rust
Array4F::from_shape_fn((batch_size, layer_config.out_channels, new_height, new_width), |(b, out_c, h, w)| {
    let h_offset = h * stride;
    let w_offset = w * stride;
    let mut result = 0.0;
    for kh in 0..*kernel_size {
        for kw in 0..*kernel_size {
            for in_c in 0..layer_config.in_channels {
                let result_h = h_offset + kh - padding;
                let result_w = w_offset + kw - padding;

                let i = if result_h < 0 || result_w < 0 || result_h >= height || result_w >= height {
                    0.0
                } else {
                    inputs[(b, in_c, h_offset + kh, w_offset + kw)];
                };
                result += i * kernel[(out_c, in_c, kh, kw)];
            }
        }
    }
    result
})
 */