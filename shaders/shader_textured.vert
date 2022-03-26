#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in mat4 in_model_matrix;
layout (location = 5) in mat4 in_inverse_model_matrix;

layout (set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view_matrix;
    mat4 projection_matrix;
} ubo;

layout (location = 0) out vec3 out_color;

void main() {
    vec4 world_pos = in_model_matrix * vec4(in_position, 1.0);
    gl_Position = ubo.projection_matrix * ubo.view_matrix * world_pos;
    out_color = vec3(1.0, 1.0, 0.5);
}