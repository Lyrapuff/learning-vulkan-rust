use ash::vk;
use gpu_allocator::vulkan::Allocator;
use nalgebra as na;
use crate::engine::allocator::VkAllocator;
use crate::engine::buffer::EngineBuffer;

pub struct DirectionalLight {
    pub direction: na::Vector3<f32>,
    pub illuminance: [f32; 3],
}

pub struct PointLight {
    pub position: na::Point3<f32>,
    pub luminous_flux: [f32; 3],
}

pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
}

impl From<PointLight> for Light {
    fn from(p: PointLight) -> Self {
        Light::Point(p)
    }
}

impl From<DirectionalLight> for Light {
    fn from(d: DirectionalLight) -> Self {
        Light::Directional(d)
    }
}

pub struct LightManager {
    directional_lights: Vec<DirectionalLight>,
    point_lights: Vec<PointLight>,
}

impl Default for LightManager {
    fn default() -> Self {
        LightManager {
            directional_lights: vec![],
            point_lights: vec![],
        }
    }
}

impl LightManager {
    pub fn add_light<T: Into<Light>>(&mut self, l: T) {
        use Light::*;

        match l.into() {
            Directional(dl) => {
                self.directional_lights.push(dl);
            },
            Point(pl) => {
                self.point_lights.push(pl);
            }
        }
    }

    pub fn update_buffer(
        &self,
        device: &ash::Device,
        allocator: &mut VkAllocator,
        buffer: &mut EngineBuffer,
        descriptor_sets_light: &mut [vk::DescriptorSet],
    ) -> Result<(), gpu_allocator::AllocationError> {
        let mut data: Vec<f32> = vec![];

        data.push(self.directional_lights.len() as f32);
        data.push(self.point_lights.len() as f32);
        data.push(0.0);
        data.push(0.0);

        for dl in &self.directional_lights {
            data.push(dl.direction.x);
            data.push(dl.direction.y);
            data.push(dl.direction.z);
            data.push(0.0);
            data.push(dl.illuminance[0]);
            data.push(dl.illuminance[1]);
            data.push(dl.illuminance[2]);
            data.push(0.0);
        }

        for pl in &self.point_lights {
            data.push(pl.position.x);
            data.push(pl.position.y);
            data.push(pl.position.z);
            data.push(0.0);
            data.push(pl.luminous_flux[0]);
            data.push(pl.luminous_flux[1]);
            data.push(pl.luminous_flux[2]);
            data.push(0.0);
        }

        let old_size = buffer.size_in_bytes;

        buffer.fill(allocator, &data)?;

        if old_size != buffer.size_in_bytes {
            for desc_set in descriptor_sets_light {
                let buffer_infos = [vk::DescriptorBufferInfo {
                    buffer: buffer.buffer,
                    offset: 0,
                    range: 4 * data.len() as u64,
                }];

                let desc_sets_write = [vk::WriteDescriptorSet::builder()
                    .dst_set(*desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos)
                    .build()];

                unsafe {
                    device.update_descriptor_sets(&desc_sets_write, &[])
                };
            }
        }

        Ok(())
    }
}