#version 450
layout(local_size_x = 8, local_size_y = 2, local_size_z = 2) in;
layout(set = 0, binding = 0) buffer ResultData {
    float data[];
} result_data;

layout(set = 0, binding = 1) buffer InputData {
    float data[];
} input_data;

layout(constant_id = 0) const uint in_channels = 0;
layout(constant_id = 1) const uint size = 0;
layout(constant_id = 2) const uint stride = 0;
layout(constant_id = 3) const uint padding = 0;

layout(constant_id = 4) const uint input_width = 0;
layout(constant_id = 5) const uint input_height = 0;

layout(constant_id = 6) const uint out_width = 0;
layout(constant_id = 7) const uint out_height = 0;

void main() {
    const uint b_and_in_c = gl_GlobalInvocationID.x;
    const uint b = b_and_in_c / in_channels;
    const uint c = b_and_in_c % in_channels;
    const uint h = gl_GlobalInvocationID.y;
    const uint w = gl_GlobalInvocationID.z;

    const uint h_offset = h * stride;
    const uint w_offset = w * stride;

    float result = -9999999999.0;

    const uint inputs_section_0 = in_channels * input_height * input_width;
    const uint inputs_section_1 = input_height * input_width;
    const uint inputs_section_2 = input_width;

    for (uint kh = 0; kh < size; kh++) {
        for (uint kw = 0; kw < size; kw++) {
            const uint result_h = h_offset + kh - padding;
            const uint result_w = w_offset + kw - padding;
            const float i = input_data.data[b*inputs_section_0 + c*inputs_section_1 + result_h*inputs_section_2 + result_w];
            result = max(result, i * float(result_h < input_height) * float(result_w < input_width));
        }
    }

    const uint index = b*in_channels*out_height*out_width + c*out_height*out_width + h*out_width + w;
    result_data.data[index] = result;
}