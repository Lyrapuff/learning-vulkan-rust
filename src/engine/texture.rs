use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use winit::window::CursorIcon::Default;
use crate::engine::allocator::VkAllocator;

pub struct Texture {
    image: image::RgbaImage,
    vk_image: vk::Image,
    allocation: Allocation,
}

impl Texture {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        device: &ash::Device,
        allocator: &mut VkAllocator
    ) -> Self {
        let image = image::open(path)
                .map(|img| img.to_rgba8())
                .expect("Failed to open image");

        let img_create_info = vk::ImageCreateInfo::builder();

        let (vk_image, allocation) = allocator.allocate_image(
            &img_create_info,
            gpu_allocator::MemoryLocation::GpuOnly,
            false
        ).unwrap();

        Texture {
            image,
            vk_image,
            allocation,
        }
    }
}