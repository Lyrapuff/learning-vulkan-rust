use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use super::surface::EngineSurface;
use super::queue_families::QueueFamilies;

pub struct EngineSwapchain {
    pub loader: ash::extensions::khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub depth_image: vk::Image,
    pub depth_image_allocation: Allocation,
    pub depth_image_view: vk::ImageView,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub image_available: Vec<vk::Semaphore>,
    pub rendering_finished: Vec<vk::Semaphore>,
    pub may_begin_drawing: Vec<vk::Fence>,
    pub amount_of_images: u32,
    pub  current_image: usize,
}

impl EngineSwapchain {
    pub fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        surfaces: &EngineSurface,
        queue_families: &QueueFamilies,
        allocator: &mut Allocator
    ) -> Result<EngineSwapchain, vk::Result> {
        let surface_capabilities = surfaces.capabilities(physical_device)?;
        let _surface_present_modes = surfaces.present_modes(physical_device)?;
        let surface_formats = surfaces.formats(physical_device)?;

        let format = surface_formats[0];
        let extent = surface_capabilities.current_extent;

        let extent3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };

        let queue_families = [queue_families.graphics_index.unwrap()];

        // Depth image creation & allocation:

        let depth_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(extent3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families);

        let depth_image = unsafe {
            device.create_image(&depth_image_info, None)?
        };

        let requirements = unsafe {
            device.get_image_memory_requirements(depth_image)
        };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "Depth Texture",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
        }).unwrap();

        unsafe {
            device.bind_image_memory(depth_image, allocation.memory(), allocation.offset())
        }?;

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(*subresource_range);

        let depth_image_view = unsafe {
            device.create_image_view(&image_view_create_info, None)
        }?;

        // Swapchain creation:

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surfaces.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count)
            )
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let amount_of_images = swapchain_images.len() as u32;

        let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());

        for image in &swapchain_images {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(*subresource_range);

            let image_view = unsafe {
                device.create_image_view(&image_view_create_info, None)?
            };

            swapchain_image_views.push(image_view);
        }

        let mut image_available = vec![];
        let mut rendering_finished = vec![];
        let mut may_begin_drawing = vec![];

        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..amount_of_images {
            let semaphore_available = unsafe {
                device.create_semaphore(&semaphore_info, None)?
            };

            let semaphore_finished = unsafe {
                device.create_semaphore(&semaphore_info, None)?
            };

            image_available.push(semaphore_available);
            rendering_finished.push(semaphore_finished);

            let fence = unsafe {
                device.create_fence(&fence_info, None)?
            };

            may_begin_drawing.push(fence);
        }

        Ok(EngineSwapchain {
            loader: swapchain_loader,
            swapchain,
            images: swapchain_images,
            image_views: swapchain_image_views,
            depth_image,
            depth_image_allocation: allocation,
            depth_image_view,
            framebuffers: vec![],
            surface_format: format,
            extent,
            amount_of_images,
            current_image: 0,
            image_available,
            rendering_finished,
            may_begin_drawing
        })
    }

    pub fn create_framebuffers(
        &mut self,
        device: &ash::Device,
        render_pass: vk::RenderPass
    ) -> Result<(), vk::Result> {
        for image_view in &self.image_views {
            let image_view = [*image_view, self.depth_image_view];

            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&image_view)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);

            let framebuffer = unsafe {
                device.create_framebuffer(&framebuffer_info, None)?
            };

            self.framebuffers.push(framebuffer);
        }

        Ok(())
    }

    pub fn calculate_current_image(&mut self) {
        self.current_image = (self.current_image + 1) % self.amount_of_images as usize;
    }

    pub unsafe fn cleanup(&mut self, device: &ash::Device) {
        device.destroy_image_view(self.depth_image_view, None);
        device.destroy_image(self.depth_image, None);

        for fence in &self.may_begin_drawing {
            device.destroy_fence(*fence, None);
        }

        for semaphore in &self.image_available {
            device.destroy_semaphore(*semaphore, None);
        }

        for semaphore in &self.rendering_finished {
            device.destroy_semaphore(*semaphore, None);
        }

        for fb in &self.framebuffers {
            device.destroy_framebuffer(*fb, None);
        }

        for iv in &self.image_views {
            device.destroy_image_view(*iv, None);
        }

        self.loader.destroy_swapchain(self.swapchain, None)
    }
}