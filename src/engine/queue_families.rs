use ash::vk;
use super::surface::EngineSurface;

pub struct QueueFamilies {
    pub graphics_index: Option<u32>,
    pub transfer_index: Option<u32>,
}

impl QueueFamilies {
    pub fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surfaces: &EngineSurface,
    ) -> Result<QueueFamilies, vk::Result> {
        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };

        let mut graphics_index = None;
        let mut transfer_index = None;

        for (i, family) in queue_family_properties.iter().enumerate() {
            if family.queue_count > 0 {
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && unsafe {
                    surfaces.surface_loader.get_physical_device_surface_support(physical_device, i as u32, surfaces.surface)?
                }
                {
                    graphics_index = Some(i as u32);
                }

                if family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    if transfer_index.is_none() || !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        transfer_index = Some(i as u32);
                    }
                }
            }
        }

        Ok(QueueFamilies {
            graphics_index: graphics_index,
            transfer_index: transfer_index,
        })
    }
}