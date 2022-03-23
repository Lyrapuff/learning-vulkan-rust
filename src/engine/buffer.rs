use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};

pub struct EngineBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size_in_bytes: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_usage: gpu_allocator::MemoryLocation,
}

impl EngineBuffer {
    pub fn new(
        allocator: &mut Allocator,
        device: &ash::Device,
        size_in_bytes: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: gpu_allocator::MemoryLocation
    ) -> Result<EngineBuffer, gpu_allocator::AllocationError> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size_in_bytes)
            .usage(usage);

        let buffer = unsafe {
            device.create_buffer(&buffer_info, None).unwrap()
        };

        let requirements = unsafe {
            device.get_buffer_memory_requirements(buffer)
        };

        let allocation = allocator.allocate(
            &AllocationCreateDesc {
                name: "Vertex Buffer",
                requirements,
                location: memory_usage,
                linear: true
            }
        ).unwrap();

        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap();
        }

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
        allocator: &mut Allocator,
        device: &ash::Device,
        data: &[T],
    ) -> Result<(), gpu_allocator::AllocationError> {
        let bytes_to_write = (data.len() * std::mem::size_of::<T>()) as u64;

        if bytes_to_write > self.size_in_bytes {
            unsafe {
                allocator.free(self.allocation.take().unwrap()).unwrap();
                device.destroy_buffer(self.buffer, None);
            }

            let new_buffer = EngineBuffer::new(
                allocator,
                device,
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
        allocator: &mut Allocator,
        device: &ash::Device,
    ) {
        allocator.free(self.allocation.take().unwrap()).unwrap();
        device.destroy_buffer(self.buffer, None);
    }
}