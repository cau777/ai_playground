#version 450
layout(local_size_x = 8, local_size_y = 2, local_size_z = 2) in;

layout(set = 0, binding = 0) buffer ResultData {
    float data[];
} result;

layout(set = 0, binding = 1) buffer InputData {
    float data[];
} inputs;

layout(set = 0, binding = 2) buffer PrevData {
    float data[];
} prev_inputs;

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

    float new_value = 0x0;

    for (uint c = 0; c < in_channels; c++) {
        const uint input_index = b*in_channels*input_height*input_width + c*input_height*input_width + h*input_width + w;

        if (abs(inputs.data[input_index] - prev_inputs.data[input_index]) > 0.0001) {
            new_value = flag;
            break;
        }
    }

    result.data[b*input_height*input_width + h*input_width + w] = new_value;
}