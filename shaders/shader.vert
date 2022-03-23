#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in mat4 in_model_matrix;
layout (location = 5) in vec3 in_color;

layout (set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view_matrix;
    mat4 projection_matrix;
} ubo;

layout (location = 0) out vec4 out_color;

void main() {
    gl_Position = ubo.projection_matrix * ubo.view_matrix * in_model_matrix * vec4(in_position, 1.0);
    out_color = vec4(in_color, 1.0);
}
