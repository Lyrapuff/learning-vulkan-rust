#version 450

layout (location = 0) in vec2 in_uv;

layout (set = 1, binding = 0) uniform sampler2D texture_sampler;

layout (location = 0) out vec4 out_color;

void main() {
    out_color = texture(texture_sampler, in_uv);
}