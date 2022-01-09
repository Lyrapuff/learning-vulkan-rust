#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in float in_size;
layout(location = 2) in vec4 in_color;

layout(location = 0) out vec4 out_color;

void main() {
    gl_PointSize = in_size;
    gl_Position = in_position;
    out_color = in_color;
}
