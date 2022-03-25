#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in mat4 in_model_matrix;
layout (location = 6) in mat4 in_inverse_model_matrix;
layout (location = 10) in vec3 in_color;
layout (location = 11) in float in_metallic;
layout (location = 12) in float in_roughness;

layout (set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view_matrix;
    mat4 projection_matrix;
} ubo;

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec4 out_world_pos;
layout (location = 3) out vec3 out_camera_pos;
layout (location = 4) out float out_metallic;
layout (location = 5) out float out_roughness;

void main() {
    out_world_pos = in_model_matrix * vec4(in_position, 1.0);

    gl_Position = ubo.projection_matrix * ubo.view_matrix * out_world_pos;

    out_normal = transpose(mat3(in_inverse_model_matrix)) * in_normal;

    out_color = in_color;

    out_camera_pos =
        - ubo.view_matrix[3][0] * vec3 (ubo.view_matrix[0][0],ubo.view_matrix[1][0],ubo.view_matrix[2][0])
        - ubo.view_matrix[3][1] * vec3 (ubo.view_matrix[0][1],ubo.view_matrix[1][1],ubo.view_matrix[2][1])
        - ubo.view_matrix[3][2] * vec3 (ubo.view_matrix[0][2],ubo.view_matrix[1][2],ubo.view_matrix[2][2]);

    out_metallic = in_metallic;
    out_roughness = in_roughness;
}
