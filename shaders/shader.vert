#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in mat4 in_model_matrix;
layout(location = 5) in vec3 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    gl_Position = in_model_matrix * vec4(in_position, 1.0);
    out_color = vec4(in_color, 1.0);
}
