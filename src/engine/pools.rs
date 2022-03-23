use ash::vk;
use super::queue_families::QueueFamilies;

pub struct Pools {
    pub command_pool_graphics: vk::CommandPool,
    pub command_pool_transfer: vk::CommandPool,
}

impl Pools {
    pub fn init (
        device: &ash::Device,
        queue_families: &QueueFamilies,
    ) -> Result<Pools, vk::Result> {
        let graphics_command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.graphics_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool_graphics = unsafe {
            device.create_command_pool(&graphics_command_pool_info, None)
        }?;

        let transfer_command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.transfer_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool_transfer = unsafe {
            device.create_command_pool(&transfer_command_pool_info, None)
        }?;

        Ok(Pools {
            command_pool_graphics,
            command_pool_transfer,
        })
    }

    pub fn create_command_buffers(
        &self,
        device: &ash::Device,
        amount: usize
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool_graphics)
            .command_buffer_count(amount as u32);

        unsafe {
            device.allocate_command_buffers(&command_buffer_allocate_info)
        }
    }

    pub fn cleanup(&self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool_graphics, None);
            device.destroy_command_pool(self.command_pool_transfer, None);
        }
    }
}