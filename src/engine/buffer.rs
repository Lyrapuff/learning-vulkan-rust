use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use crate::engine::allocator::VkAllocator;

pub struct EngineBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size_in_bytes: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_usage: gpu_allocator::MemoryLocation,
}

impl EngineBuffer {
    pub fn new(
        allocator: &mut VkAllocator,
        size_in_bytes: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: gpu_allocator::MemoryLocation
    ) -> Result<EngineBuffer, gpu_allocator::AllocationError> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size_in_bytes)
            .usage(usage);

        let (buffer, allocation) = allocator.allocate_buffer(
            &buffer_info,
            memory_usage,
            true
        ).unwrap();

        Ok(EngineBuffer {
            buffer,
            allocation: Some(allocation),
            size_in_bytes,
            usage,
            memory_usage,
        })
    }

    pub fn fill<T: Sized>(
        &mut self,
        allocator: &mut VkAllocator,
        data: &[T],
    ) -> Result<(), gpu_allocator::AllocationError> {
        let bytes_to_write = (data.len() * std::mem::size_of::<T>()) as u64;

        if bytes_to_write > self.size_in_bytes {
            unsafe {
                self.cleanup(allocator);
            }

            let new_buffer = EngineBuffer::new(
                allocator,
                bytes_to_write,
                self.usage,
                self.memory_usage
            )?;

            *self = new_buffer;
        }

        if let Some(allocation) = &self.allocation {
            let data_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut T;

            unsafe {
                data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
            }
        }

        Ok(())
    }

    pub unsafe fn cleanup(
        &mut self,
        allocator: &mut VkAllocator,
    ) {
        let destroyer = |device: &ash::Device| device.destroy_buffer(self.buffer, None);
        allocator.free(self.allocation.take().unwrap(), &destroyer);
    }
}