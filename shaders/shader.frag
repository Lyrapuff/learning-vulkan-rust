#version 450

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 direction_to_light = normalize(vec3(-1.0, -1.0, 0.0));

    out_color = in_color * max(dot(in_normal, direction_to_light), 0);
}
