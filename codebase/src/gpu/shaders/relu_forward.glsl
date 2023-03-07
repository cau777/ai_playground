#version 450
layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer ResultData {
    float data[];
} result_data;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    result_data.data[index] = max(0.0, result_data.data[index]);
}