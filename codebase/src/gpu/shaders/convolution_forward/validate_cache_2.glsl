#version 450
layout(local_size_x = 8, local_size_y = 2, local_size_z = 2) in;

layout(set = 0, binding = 0) buffer ResultData {
    float data[];
} result;

layout(set = 0, binding = 1) buffer InputData {
    float data[];
} inputs;

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
    const uint b = gl_GlobalInvocationID.x;
    const uint h = gl_GlobalInvocationID.y;
    const uint w = gl_GlobalInvocationID.z;
    const float flag = uintBitsToFloat(1);

    const uint h_offset = h * stride;
    const uint w_offset = w * stride;
    float new_value = 0x0;

    for (uint kh = 0; kh < kernel_size; kh++) {
        for (uint kw = 0; kw < kernel_size; kw++) {
            const uint input_h = h_offset + kh - padding;
            const uint input_w = w_offset + kw - padding;

            const bool in_bounds = (input_h < input_height) && (input_w < input_width);
            const uint invalid = uint(in_bounds) & floatBitsToUint(inputs.data[b*input_height*input_width + input_h*input_width + input_w]);

//            new_value = float(floatBitsToUint(inputs.data[b*input_height*input_width + input_h*input_width + input_w]));
            if (invalid == 0x1) {
                new_value = flag;
                break;
            }
        }
    }

//    result.data[b*out_height*out_width + h*out_width + w] = new_value * 0.000001
//    + inputs.data[b*input_height*input_width + 3*input_width + 3];
//    result.data[b*out_height*out_width + h*out_width + w] = new_value * 0.000001+ float(kernel_size);
    result.data[b*out_height*out_width + h*out_width + w] = new_value;
}