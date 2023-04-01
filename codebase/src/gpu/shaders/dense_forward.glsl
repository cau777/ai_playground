#version 450
layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer ResultData {
    float data[];
} result_data;

layout(set = 0, binding = 1) buffer WeightsData {
    float data[];
} weights_data;

layout(set = 0, binding = 2) buffer BiasesData {
    float data[];
} biases_data;

layout(set = 0, binding = 3) buffer InputData {
    float data[];
} input_data;

layout(constant_id = 0) const uint in_values = 0;
layout(constant_id = 1) const uint out_values = 0;

void main() {
    const uint b = gl_GlobalInvocationID.x;
    const uint o = gl_GlobalInvocationID.y;

    const uint input_offset = b * in_values;
    const uint weights_offset = o * in_values;

    float result = 0.0;
    for (uint i = 0; i < in_values; i++) {
        result += weights_data.data[weights_offset + i] * input_data.data[input_offset + i];
    }

    result_data.data[b * out_values + o] = result + biases_data.data[o];
}