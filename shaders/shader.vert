#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_position_offset;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    gl_Position = vec4(in_position + in_position_offset, 1.0);
    out_color = vec4(in_color, 1.0);
}
