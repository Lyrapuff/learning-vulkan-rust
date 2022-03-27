use std::error::Error;
use std::mem::ManuallyDrop;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use ash::{vk};
use gpu_allocator::{AllocationError, MemoryLocation};

pub struct VkAllocator {
    device: ash::Device,
    allocator: ManuallyDrop<Allocator>,
}

impl VkAllocator {
    pub fn new(info: &AllocatorCreateDesc) -> VkAllocator {
        let allocator = Allocator::new(&info).unwrap();

        VkAllocator {
            device: info.device.clone(),
            allocator: ManuallyDrop::new(allocator)
        }
    }

    pub fn allocate(&mut self, info: &AllocationCreateDesc) -> Result<Allocation, AllocationError> {
        self.allocator.allocate(info)
    }

    // Maybe create fns like free_image and free_buffer for convenience
    pub fn free(
        &mut self,
        allocation: Allocation,
        destroyer: &dyn Fn(&ash::Device) -> ()
    ) {
        self.allocator.free(allocation).unwrap();
        destroyer(&self.device);
    }

    pub fn allocate_image(
        &mut self,
        image_info: &vk::ImageCreateInfo,
        location: MemoryLocation,
        linear: bool,
    ) -> Result<(vk::Image, Allocation), Box<dyn Error>> {
        let image = unsafe {
            self.device.create_image(image_info, None)
        }?;

        let requirements = unsafe {
            self.device.get_image_memory_requirements(image)
        };

        let allocation = self.allocate(&AllocationCreateDesc {
            name: "VkAllocator Image",
            location,
            requirements,
            linear,
        })?;

        unsafe {
            self.device.bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        Ok((image, allocation))
    }

    pub fn allocate_buffer(
        &mut self,
        buffer_info: &vk::BufferCreateInfo,
        location: MemoryLocation,
        linear: bool,
    ) -> Result<(vk::Buffer, Allocation), Box<dyn Error>> {
        let buffer = unsafe {
            self.device.create_buffer(&buffer_info, None)
        }?;

        let requirements = unsafe {
            self.device.get_buffer_memory_requirements(buffer)
        };

        let allocation = self.allocate(
            &AllocationCreateDesc {
                name: "VkAllocator Buffer",
                requirements,
                location,
                linear,
            }
        )?;

        unsafe {
            self.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }?;

        Ok((buffer, allocation))
    }

    pub fn cleanup(&mut self) {
       unsafe {
           ManuallyDrop::drop(&mut self.allocator);
       }
    }
}