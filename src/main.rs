use std::collections::HashMap;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;

use ash::{Device, Entry, Instance, vk};
use ash::extensions::khr::Swapchain;
use ash::vk::{Buffer, CommandBuffer, CommandPool, Extent2D, Framebuffer, Handle, Image, ImageView, MemoryRequirements, Offset2D, PhysicalDevice, PhysicalDeviceProperties, Pipeline, PipelineLayout, Queue, Rect2D, RenderPass, Viewport};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};

use winit::event_loop::EventLoop;
use winit::window::Window;
use winit::event::{Event, WindowEvent};

use nalgebra as na;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void
) -> vk::Bool32 {
    let message = CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();

    println!("[Debug][{}][{}] {:?}", severity, ty, message);

    vk::FALSE
}

struct VulkanEngine {
    window: Window,
    entry: Entry,
    instance: Instance,
    debug: ManuallyDrop<EngineDebug>,
    surfaces: ManuallyDrop<EngineSurface>,
    physical_device: PhysicalDevice,
    physical_device_properties: PhysicalDeviceProperties,
    queue_families: QueueFamilies,
    queues: Queues,
    device: Device,
    swapchain: EngineSwapchain,
    render_pass: RenderPass,
    pipeline: EnginePipeline,
    pools: Pools,
    graphics_command_buffers: Vec<CommandBuffer>,
    allocator: ManuallyDrop<Allocator>,
    models: Vec<Model<[f32; 3], InstanceData>>,
}

impl VulkanEngine {
    pub fn init(window: Window) -> Result<VulkanEngine, vk::Result> {
        let entry = Entry::linked();

        let layer_names = vec!["VK_LAYER_KHRONOS_validation"];

        let instance = Self::init_instance(&entry, &layer_names)?;

        let debug = EngineDebug::init(&entry, &instance)?;

        let surfaces = EngineSurface::init(&window, &entry, &instance)?;

        let (physical_device, physical_device_properties) = Self::init_physical_device(&instance)?;

        let queue_families = QueueFamilies::init(&instance, physical_device, &surfaces)?;

        let (device, queues) = Self::init_device_queues(&instance, physical_device, &queue_families, &layer_names)?;

        let mut allocator = Allocator::new(
            &AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false
            }
        ).unwrap();

        let mut swapchain = EngineSwapchain::init(
            &instance,
            physical_device,
            &device,
            &surfaces,
            &queue_families,
            &mut allocator
        )?;

        let render_pass = Self::init_render_pass(&device, physical_device, &surfaces)?;

        swapchain.create_framebuffers(&device, render_pass)?;

        let pipeline = EnginePipeline::init(&device, &swapchain, render_pass)?;

        let pools = Pools::init(&device, &queue_families)?;
        let command_buffers = pools.create_command_buffers(&device, swapchain.framebuffers.len())?;

        let engine = VulkanEngine {
            window,
            entry,
            instance,
            debug: std::mem::ManuallyDrop::new(debug),
            surfaces: std::mem::ManuallyDrop::new(surfaces),
            physical_device,
            physical_device_properties,
            queue_families,
            queues,
            device,
            swapchain,
            render_pass,
            pipeline,
            pools,
            graphics_command_buffers: command_buffers,
            allocator: ManuallyDrop::new(allocator),
            models: vec![],
        };

        engine.fill_command_buffers(&engine.models);

        Ok(engine)
    }

    fn init_instance(
        entry: &Entry,
        layer_names: &[&str],
    ) -> Result<Instance, vk::Result> {
        let app_name = CString::new("Vulkan Engine").unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_1);

        let layer_names: Vec<CString> = layer_names
            .iter()
            .map(|&ln| CString::new(ln).unwrap())
            .collect();
        let layer_name_pts: Vec<*const i8> = layer_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();
        let extension_name_pts: Vec<*const i8> = vec![
            ash::extensions::ext::DebugUtils::name().as_ptr(),
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::XlibSurface::name().as_ptr(),
        ];

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layer_name_pts)
            .enabled_extension_names(&extension_name_pts);

        unsafe {
            entry.create_instance(&instance_create_info, None)
        }
    }

    fn init_physical_device(
        instance: &Instance,
    ) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties), vk::Result> {
        let phys_devs = unsafe {
            instance.enumerate_physical_devices()?
        };

        let p = phys_devs[0];

        let properties = unsafe {
            instance.get_physical_device_properties(p)
        };

        Ok((p, properties))
    }

    fn init_device_queues(
        instance: &Instance,
        physical_device: PhysicalDevice,
        queue_families: &QueueFamilies,
        layer_names: &[&str],
    ) -> Result<(Device, Queues), vk::Result> {
        let layer_names: Vec<CString> = layer_names
            .iter()
            .map(|&ln| CString::new(ln).unwrap())
            .collect();
        let layer_name_pts: Vec<*const i8> = layer_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        // possible problem: index 0 and 1 are the same, if this is the case we need only one DeviceQueueCreateInfo
        let priorities = [1.0f32];
        let queue_infos = [
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_families.graphics_index.unwrap())
                .queue_priorities(&priorities)
                .build(),
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_families.transfer_index.unwrap())
                .queue_priorities(&priorities)
                .build(),
        ];

        let device_extensions_name_pts: Vec<*const i8> = vec![
            ash::extensions::khr::Swapchain::name().as_ptr()
        ];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions_name_pts)
            .enabled_layer_names(&layer_name_pts);

        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)?
        };

        let graphics_queue = unsafe {
            device.get_device_queue(queue_families.graphics_index.unwrap(), 0)
        };
        let transfer_queue = unsafe {
            device.get_device_queue(queue_families.transfer_index.unwrap(), 0)
        };

        Ok((device, Queues {
            graphics: graphics_queue,
            transfer: transfer_queue
        }))
    }

    fn init_render_pass(
        device: &Device,
        physical_device: PhysicalDevice,
        surfaces: &EngineSurface
    ) -> Result<RenderPass, vk::Result> {
        let attachments = [
            vk::AttachmentDescription::builder()
                .format(
                    surfaces.formats(physical_device)?
                        .first()
                        .unwrap()
                        .format
                )
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
            vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build()
        ];

        let color_attachment_refs = [
            vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }
        ];

        let depth_attachment_refs = [
            vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            }
        ];

        let subpasses = [
            vk::SubpassDescription::builder()
                .color_attachments(&color_attachment_refs)
                .depth_stencil_attachment(&depth_attachment_refs[0])
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .build()
        ];

        let subpass_dependencies = [
            vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_subpass(0)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                )
                .build()
        ];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies);

        unsafe {
            device.create_render_pass(&render_pass_info, None)
        }
    }

    fn update_command_buffer(&mut self, index: usize) -> Result<(), vk::Result> {
        let command_buffer = self.graphics_command_buffers[index];
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();

        unsafe {
            self.device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;
        }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.08, 1.0],
                }
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                }
            }
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain.framebuffers[index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D {
                    x: 0,
                    y: 0,
                },
                extent: self.swapchain.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE
            );

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline
            );

            for m in &self.models {
                m.draw(&self.device, command_buffer);
            }

            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn fill_command_buffers(&self, models: &[Model<[f32; 3], InstanceData>]) {
        for (i, &command_buffer) in self.graphics_command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();

            unsafe {
                self.device.begin_command_buffer(command_buffer, &command_buffer_begin_info);
            }

            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    }
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    }
                }
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(self.render_pass)
                .framebuffer(self.swapchain.framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D {
                        x: 0,
                        y: 0,
                    },
                    extent: self.swapchain.extent
                })
                .clear_values(&clear_values);

            unsafe {
                self.device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE
                );

                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline
                );

                //draw models
                for model in models {
                    model.draw(&self.device, command_buffer);
                }

                self.device.cmd_end_render_pass(command_buffer);

                self.device.end_command_buffer(command_buffer);
            }
        }
    }
}

impl Drop for VulkanEngine{
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait?");

            for m in &mut self.models {
                if let Some(vb) = &mut m.vertex_buffer {
                    vb.cleanup(&mut self.allocator, &self.device);
                }
                if let Some(ib) = &mut m.instance_buffer {
                    ib.cleanup(&mut self.allocator, &self.device);
                }
            }

            ManuallyDrop::drop(&mut self.allocator);

            self.pools.cleanup(&self.device);

            self.pipeline.cleanup(&self.device);

            self.device.destroy_render_pass(self.render_pass, None);

            self.swapchain.cleanup(&self.device);

            ManuallyDrop::drop(&mut self.surfaces);

            ManuallyDrop::drop(&mut self.debug);

            self.device.destroy_device(None);

            self.instance.destroy_instance(None);
        }
    }
}

struct EngineDebug {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl EngineDebug {
    fn init(entry: &Entry, instance: &Instance) -> Result<EngineDebug, vk::Result> {
        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        
        let messenger = unsafe {
            loader.create_debug_utils_messenger(&debug_create_info, None)?
        };

        Ok(EngineDebug {
            loader,
            messenger
        })
    }
}

impl Drop for EngineDebug {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_debug_utils_messenger(self.messenger, None)
        }
    }
}

// os specific (in my case linux), will use ash_window crate later, probably, if anyone will care
struct EngineSurface {
    xlib_surface_loader: ash::extensions::khr::XlibSurface,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
}

impl EngineSurface {
    fn init(
        window: &winit::window::Window,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<EngineSurface, vk::Result> {
        let xlib_surface_loader = ash::extensions::khr::XlibSurface::new(&entry, &instance);
        let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

        Ok(EngineSurface {
            xlib_surface_loader,
            surface,
            surface_loader,
        })
    }

    fn capabilities(
        &self,
        physical_device: vk::PhysicalDevice
    ) -> Result<vk::SurfaceCapabilitiesKHR, vk::Result> {
        unsafe {
            self.surface_loader.get_physical_device_surface_capabilities(physical_device, self.surface)
        }
    }

    fn present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, vk::Result> {
        unsafe {
            self.surface_loader.get_physical_device_surface_present_modes(physical_device, self.surface)
        }
    }

    fn formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, vk::Result> {
        unsafe {
            self.surface_loader.get_physical_device_surface_formats(physical_device, self.surface)
        }
    }

    fn physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: usize,
    ) -> Result<bool, vk::Result> {
        unsafe {
            self.surface_loader.get_physical_device_surface_support(
                physical_device,
                queue_family_index as u32,
                self.surface,
            )
        }
    }
}

impl Drop for EngineSurface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

struct QueueFamilies {
    graphics_index: Option<u32>,
    transfer_index: Option<u32>,
}

impl QueueFamilies {
    fn init(
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

struct Queues {
    graphics: Queue,
    transfer: Queue,
}

struct EngineSwapchain {
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<Image>,
    image_views: Vec<ImageView>,
    depth_image: vk::Image,
    depth_image_allocation: Allocation,
    depth_image_view: ImageView,
    framebuffers: Vec<Framebuffer>,
    surface_format: vk::SurfaceFormatKHR,
    extent: Extent2D,
    image_available: Vec<vk::Semaphore>,
    rendering_finished: Vec<vk::Semaphore>,
    may_begin_drawing: Vec<vk::Fence>,
    amount_of_images: u32,
    current_image: usize,
}

impl EngineSwapchain {
    fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        surfaces: &EngineSurface,
        queue_families: &QueueFamilies,
        allocator: &mut Allocator
    ) -> Result<EngineSwapchain, vk::Result> {
        let surface_capabilities = surfaces.capabilities(physical_device)?;
        let surface_present_modes = surfaces.present_modes(physical_device)?;
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

    fn create_framebuffers(
        &mut self,
        device: &Device,
        render_pass: RenderPass
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

    fn calculate_current_image(&mut self) {
        self.current_image = (self.current_image + 1) % self.amount_of_images as usize;
    }

    unsafe fn cleanup(&mut self, device: &Device) {
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

struct EnginePipeline {
    pipeline: Pipeline,
    layout: PipelineLayout,
}

impl EnginePipeline {
    fn init(
        device: &Device,
        swapchain: &EngineSwapchain,
        render_pass: RenderPass
    ) -> Result<EnginePipeline, vk::Result> {
        let vertex_shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(
                vk_shader_macros::include_glsl!("./shaders/shader.vert")
            );
        let vertex_shader_module = unsafe {
            device.create_shader_module(&vertex_shader_create_info, None)?
        };

        let fragment_shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(
                vk_shader_macros::include_glsl!("./shaders/shader.frag")
            );
        let fragment_shader_module = unsafe {
            device.create_shader_module(&fragment_shader_create_info, None)?
        };

        let entry_point = CString::new("main").unwrap();
        let vertex_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point);
        let fragment_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point);
        let shader_stages = vec![
            vertex_shader_stage.build(),
            fragment_shader_stage.build()
        ];

        let vertex_attrib_descs = [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                offset: 0,
                format: vk::Format::R32G32B32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                offset: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                offset: 16,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                offset: 32,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                offset: 48,
                format: vk::Format::R32G32B32A32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                offset: 64,
                format: vk::Format::R32G32B32_SFLOAT,
            },
        ];

        let vertex_binding_descs = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: 12,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: 76,
                input_rate: vk::VertexInputRate::INSTANCE,
            },
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_attrib_descs)
            .vertex_binding_descriptions(&vertex_binding_descs);

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewports = [
            Viewport {
                x: 0.0,
                y: 0.0,
                width: swapchain.extent.width as f32,
                height: swapchain.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        ];
        let scissors = [
            Rect2D {
                offset: Offset2D {
                    x: 0,
                    y: 0,
                },
                extent: swapchain.extent
            }
        ];

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);

        let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let colorblend_attachments = [
            vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .build(),
        ];

        let colorblend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&colorblend_attachments);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder();
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&pipeline_layout_info, None)?
        };

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampler_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&colorblend_info)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipeline = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info.build()],
                None
            ).expect("Failed to create graphics pipeline")
        }[0];

        unsafe {
            device.destroy_shader_module(fragment_shader_module, None);
            device.destroy_shader_module(vertex_shader_module, None);
        }

        Ok(EnginePipeline {
            pipeline: graphics_pipeline,
            layout: pipeline_layout,
        })
    }

    fn cleanup(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

struct Pools {
    command_pool_graphics: CommandPool,
    command_pool_transfer: CommandPool,
}

impl Pools {
    fn init (
        device: &Device,
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

    fn create_command_buffers(
        &self,
        device: &Device,
        amount: usize
    ) -> Result<Vec<CommandBuffer>, vk::Result> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool_graphics)
            .command_buffer_count(amount as u32);

        unsafe {
            device.allocate_command_buffers(&command_buffer_allocate_info)
        }
    }

    fn cleanup(&self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool_graphics, None);
            device.destroy_command_pool(self.command_pool_transfer, None);
        }
    }
}

struct EngineBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    size_in_bytes: u64,
    usage: vk::BufferUsageFlags,
    memory_usage: gpu_allocator::MemoryLocation,
}

impl EngineBuffer {
    fn new(
        allocator: &mut Allocator,
        device: &Device,
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

    fn fill<T: Sized>(
        &mut self,
        allocator: &mut Allocator,
        device: &Device,
        data: &[T]
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

    unsafe fn cleanup(
        &mut self,
        allocator: &mut Allocator,
        device: &Device,
    ) {
        allocator.free(self.allocation.take().unwrap()).unwrap();
        device.destroy_buffer(self.buffer, None);
    }
}

#[derive(Debug, Clone)]
struct InvalidHandle;

impl std::fmt::Display for InvalidHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "invalid handle")
    }
}
impl std::error::Error for InvalidHandle {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[repr(C)]
struct InstanceData {
    model_matrix: [[f32; 4]; 4],
    color: [f32; 3],
}

struct Model<V, I> {
    vertex_data: Vec<V>,
    handle_to_index: HashMap<usize, usize>,
    handles: Vec<usize>,
    instances: Vec<I>,
    first_invisible: usize,
    next_handle: usize,
    vertex_buffer: Option<EngineBuffer>,
    instance_buffer: Option<EngineBuffer>,
}

impl<V, I> Model<V, I> {
    fn get(&self, handle: usize) -> Option<&I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get(index)
        }
        else {
            None
        }
    }

    fn get_mut(&mut self, handle: usize) -> Option<&mut I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get_mut(index)
        } else {
            None
        }
    }

    fn swap_by_handle(&mut self, h1: usize, h2: usize) -> Result<(), InvalidHandle> {
        if h1 == h2 {
            return Ok(());
        }

        if let (Some(&i1), Some(&i2)) = (
            self.handle_to_index.get(&h1),
            self.handle_to_index.get(&h2)
        ) {
            self.handles.swap(i1, i2);
            self.instances.swap(i1, i2);
            self.handle_to_index.insert(i1, h2);
            self.handle_to_index.insert(i2, h1);
            Ok(())
        }
        else {
            Err(InvalidHandle)
        }
    }

    fn swap_by_index(&mut self, index1: usize, index2: usize) {
        if index1 == index2 {
            return;
        }

        let handle1 = self.handles[index1];
        let handle2 = self.handles[index2];
        self.handles.swap(index1, index2);
        self.instances.swap(index1, index2);
        self.handle_to_index.insert(index1, handle2);
        self.handle_to_index.insert(index2, handle1);
    }

    fn is_visible(&self, handle: usize) -> Result<bool, InvalidHandle> {
        if let Some(index) = self.handle_to_index.get(&handle) {
            Ok(index < &self.first_invisible)
        }
        else {
            Err(InvalidHandle)
        }
    }

    fn make_visible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                return Ok(());
            }

            self.swap_by_index(index, self.first_invisible);
            self.first_invisible += 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_invisible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index >= self.first_invisible {
                return Ok(());
            }

            self.swap_by_index(index, self.first_invisible - 1);
            self.first_invisible -= 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn insert(&mut self, element: I) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;

        let index = self.instances.len();
        self.instances.push(element);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);

        handle
    }

    fn insert_visibly(&mut self, element: I) -> usize {
        let new_handle = self.insert(element);
        self.make_visible(new_handle).ok();
        new_handle
    }

    fn remove(&mut self, handle: usize) -> Result<I, InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                self.swap_by_index(index, self.first_invisible - 1);
                self.first_invisible -= 1;
            }

            self.swap_by_index(self.first_invisible, self.instances.len() - 1);
            self.handles.pop();
            self.handle_to_index.remove(&handle);

            Ok(self.instances.pop().unwrap())
        } else {
            Err(InvalidHandle)
        }
    }

    fn update_vertex_buffer(&mut self, device: &Device, allocator: &mut Allocator) -> Result<(), gpu_allocator::AllocationError> {
        if let Some(buffer) = &mut self.vertex_buffer {
            buffer.fill(allocator, device, &self.vertex_data)?;
            Ok(())
        } else {
            let bytes = (self.vertex_data.len() * std::mem::size_of::<V>()) as u64;
            let mut buffer = EngineBuffer::new(
                allocator,
                device,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;

            buffer.fill(allocator, device, &self.vertex_data)?;
            self.vertex_buffer = Some(buffer);

            Ok(())
        }
    }

    fn update_instance_buffer(&mut self, device: &Device, allocator: &mut Allocator) -> Result<(), gpu_allocator::AllocationError> {
        if let Some(buffer) = &mut self.instance_buffer {
            buffer.fill(allocator, device, &self.instances[0..self.first_invisible])?;
            Ok(())
        } else {
            let bytes = (self.first_invisible * std::mem::size_of::<I>()) as u64;
            let mut buffer = EngineBuffer::new(
                allocator,
                device,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;

            buffer.fill(allocator, device, &self.instances[0..self.first_invisible])?;
            self.instance_buffer = Some(buffer);

            Ok(())
        }
    }

    fn draw(&self, device: &Device, command_buffer: CommandBuffer) {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if let Some(instance_buffer) = &self.instance_buffer {
                if self.first_invisible > 0 {
                    unsafe {
                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vertex_buffer.buffer],
                            &[0]
                        );

                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            1,
                            &[instance_buffer.buffer],
                            &[0]
                        );

                        device.cmd_draw(
                            command_buffer,
                            self.vertex_data.len() as u32,
                            self.first_invisible as u32,
                            0,
                            0,
                        );
                    }
                }
            }
        }
    }
}

impl Model<[f32; 3], InstanceData> {
    fn cube() -> Self {
        let lbf = [-1.0,1.0,0.0]; //lbf: left-bottom-front
        let lbb = [-1.0,1.0,1.0];
        let ltf = [-1.0,-1.0,0.0];
        let ltb = [-1.0,-1.0,1.0];
        let rbf = [1.0,1.0,0.0];
        let rbb = [1.0,1.0,1.0];
        let rtf = [1.0,-1.0,0.0];
        let rtb = [1.0,-1.0,1.0];

        Model {
            vertex_data: vec![
                lbf, lbb, rbb, lbf, rbb, rbf, //bottom
                ltf, rtb, ltb, ltf, rtf, rtb, //top
                lbf, rtf, ltf, lbf, rbf, rtf, //front
                lbb, ltb, rtb, lbb, rtb, rbb, //back
                lbf, ltf, lbb, lbb, ltf, ltb, //left
                rbf, rbb, rtf, rbb, rtb, rtf, //right
            ],
            handle_to_index: HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            instance_buffer: None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop)?;

    let mut engine = VulkanEngine::init(window)?;

    let mut cube = Model::cube();

    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.1))
            * na::Matrix4::new_scaling(0.1))
            .into(),
        color: [0.2, 0.4, 1.0],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.05, 0.05, 0.0))
            * na::Matrix4::new_scaling(0.1))
            .into(),
        color: [1.0, 1.0, 0.2],
    });
    for i in 0..10 {
        for j in 0..10 {
            cube.insert_visibly(InstanceData {
                model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(
                    i as f32 * 0.2 - 1.0,
                    j as f32 * 0.2 - 1.0,
                    0.5,
                )) * na::Matrix4::new_scaling(0.03))
                    .into(),
                color: [1.0, i as f32 * 0.07, j as f32 * 0.07],
            });
            cube.insert_visibly(InstanceData {
                model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(
                    i as f32 * 0.2 - 1.0,
                    0.0,
                    j as f32 * 0.2 - 1.0,
                )) * na::Matrix4::new_scaling(0.02))
                    .into(),
                color: [i as f32 * 0.07, j as f32 * 0.07, 1.0],
            });
        }
    }
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::from_scaled_axis(na::Vector3::new(0.0, 0.0, 1.4))
            * na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.5, 0.0))
            * na::Matrix4::new_scaling(0.1))
            .into(),
        color: [0.0, 0.5, 0.0],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.5, 0.0, 0.0))
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(0.5, 0.01, 0.01)))
            .into(),
        color: [1.0, 0.5, 0.5],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.5, 0.0))
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(0.01, 0.5, 0.01)))
            .into(),
        color: [0.5, 1.0, 0.5],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.0))
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(0.01, 0.01, 0.5)))
            .into(),
        color: [0.5, 0.5, 1.0],
    });

    cube.update_vertex_buffer(&engine.device, &mut engine.allocator).unwrap();
    cube.update_instance_buffer(&engine.device, &mut engine.allocator).unwrap();

    let models = vec![cube];
    engine.models = models;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = winit::event_loop::ControlFlow::Exit;
            }
            Event::MainEventsCleared => {
                engine.window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                engine.swapchain.calculate_current_image();

                let (image_index, _) = unsafe {
                    engine.swapchain.loader.acquire_next_image(
                        engine.swapchain.swapchain,
                        u64::MAX,
                        engine.swapchain.image_available[engine.swapchain.current_image],
                        vk::Fence::null()
                    ).expect("Failed to acquire next image")
                };

                unsafe {
                    engine.device.wait_for_fences(
                        &[engine.swapchain.may_begin_drawing[engine.swapchain.current_image]],
                        true,
                        u64::MAX
                    ).expect("Fence waiting");

                    engine.device.reset_fences(
                        &[engine.swapchain.may_begin_drawing[engine.swapchain.current_image]]
                    ).expect("Resetting fences");

                    for m in &mut engine.models {
                        m.update_instance_buffer(&engine.device, &mut engine.allocator);
                    }

                    engine.update_command_buffer(image_index as usize)
                        .expect("Failed to update command buffer");

                    let semaphores_available = [engine.swapchain.image_available[engine.swapchain.current_image]];
                    let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                    let semaphores_finished = [engine.swapchain.rendering_finished[engine.swapchain.current_image]];
                    let command_buffers = [engine.graphics_command_buffers[image_index as usize]];

                    let submit_info = [
                        vk::SubmitInfo::builder()
                            .wait_semaphores(&semaphores_available)
                            .wait_dst_stage_mask(&waiting_stages)
                            .command_buffers(&command_buffers)
                            .signal_semaphores(&semaphores_finished)
                            .build()
                    ];

                    unsafe {
                        engine.device.queue_submit(
                            engine.queues.graphics,
                            &submit_info,
                            engine.swapchain.may_begin_drawing[engine.swapchain.current_image]
                        ).expect("Queue submission");
                    }

                    let swapchains = [engine.swapchain.swapchain];
                    let indices = [image_index];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&semaphores_finished)
                        .swapchains(&swapchains)
                        .image_indices(&indices);

                    unsafe {
                        engine.swapchain.loader.queue_present(
                            engine.queues.graphics,
                            &present_info
                        ).expect("Queue presentation");
                    }
                }
            }
            _ => {}
        }
    });
}