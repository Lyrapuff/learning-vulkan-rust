#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_world_pos;
layout(location = 3) in vec3 in_camera_pos;
layout(location = 4) in float in_metallic;
layout(location = 5) in float in_roughness;

layout(location = 0) out vec4 out_color;

readonly layout (set = 1, binding = 0) buffer StorageBufferObject {
    float num_directional;
    float num_point;
    vec3 data[];
} sbo;

struct DirectionalLight {
    vec3 direction_to_light;
    vec3 irradiance;
};

struct PointLight {
    vec3 position;
    vec3 luminous_flux;
};

const float PI = 3.14159265358979323846264;

float distribution(vec3 normal,vec3 halfvector,float roughness) {
    float NdotH=dot(halfvector,normal);
    if (NdotH>0){
        float r=roughness*roughness;
        return r / (PI* (1 + NdotH*NdotH*(r-1))*(1 + NdotH*NdotH*(r-1)));
    }else{
        return 0.0;
    }
}

float geometry(vec3 light, vec3 normal, vec3 view, float roughness) {
    float NdotL=abs(dot(normal,light));
    float NdotV=abs(dot(normal,view));
    return 0.5/max(0.01,mix(2*NdotL*NdotV,NdotL+NdotV,roughness));
}

vec3 compute_radiance(vec3 irradiance, vec3 light_direction, vec3 normal, vec3 camera_dir, vec3 surface_color) {
    float n_dot_l = max(dot(normal, light_direction), 0);

    vec3 irradiance_on_surface = irradiance * n_dot_l;

    float roughness = in_roughness * in_roughness;

    vec3 F0 = mix(vec3(0.03), in_color, vec3(in_metallic));

    vec3 reflected_irradiance = (F0+(1-F0)*(1-n_dot_l)*(1-n_dot_l)*(1-n_dot_l)*(1-n_dot_l)*(1-n_dot_l))*irradiance_on_surface;
    vec3 refracted_irradiance = irradiance_on_surface - reflected_irradiance;
    vec3 refracted_not_absorbed_irradiance = refracted_irradiance * (1 - in_metallic);

    vec3 half_vector = normalize(0.5 * (camera_dir + light_direction));
    float n_dot_h = max(dot(normal, half_vector), 0);
    vec3 F = (F0 + (1 - F0)*(1-n_dot_h)*(1-n_dot_h)*(1-n_dot_h)*(1-n_dot_h)*(1-n_dot_h));

    vec3 relevant_reflection = reflected_irradiance*F*geometry(light_direction,normal,camera_dir,roughness)*distribution(normal,half_vector,roughness);

    return refracted_not_absorbed_irradiance * surface_color / PI + relevant_reflection;
}

void main() {
    vec3 normal = normalize(in_normal);
    vec3 direction_to_camera = normalize(in_camera_pos - in_world_pos);

    vec3 light = vec3(0);

    // Directional lights:

    int number_directional = int(sbo.num_directional);
    int number_point = int(sbo.num_point);

    for (int i = 0; i < number_directional; i++) {
        vec3 data1=sbo.data[2*i];
        vec3 data2=sbo.data[2*i+1];

        DirectionalLight dlight = DirectionalLight(normalize(data1),data2);

        light += compute_radiance(dlight.irradiance, dlight.direction_to_light, normal, direction_to_camera, in_color);
    }

    // Point lights:

    for (int i=0;i<number_point;i++){
        vec3 data1=sbo.data[2*i+2*number_directional];
        vec3 data2=sbo.data[2*i+1+2*number_directional];
        PointLight plight = PointLight(data1,data2);
        vec3 direction_to_light = normalize(plight.position - in_world_pos);
        float d = length(in_world_pos - plight.position);
        vec3 irradiance = plight.luminous_flux/(4*PI*d*d);

        light += compute_radiance(irradiance, direction_to_light, normal, direction_to_camera, in_color);
    }

    // Output:

    out_color = vec4(light / (1 + light), 1.0);
}
