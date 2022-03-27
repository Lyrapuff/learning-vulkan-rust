use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use crate::engine::allocator::VkAllocator;
use crate::engine::buffer::EngineBuffer;

pub struct Texture {
    pub image: image::RgbaImage,
    pub width: u32,
    pub height: u32,
    pub vk_image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
    pub sampler: vk::Sampler,
}

impl Texture {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        device: &ash::Device,
        allocator: &mut VkAllocator
    ) -> Self {
        let image = image::open(path)
            .expect("Failed to open image")
            .to_rgba8();

        let (width, height) = image.dimensions();

        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);

        let (vk_image, allocation) = allocator.allocate_image(
            &image_create_info,
            gpu_allocator::MemoryLocation::GpuOnly,
            false
        ).unwrap();

        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(vk_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            });

        let image_view = unsafe {
            device.create_image_view(&image_view_create_info, None)
        }.unwrap();

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);

        let sampler = unsafe {
            device.create_sampler(&sampler_info, None)
        }.unwrap();

        Texture {
            image,
            width,
            height,
            vk_image,
            image_view,
            allocation,
            sampler,
        }
    }
}