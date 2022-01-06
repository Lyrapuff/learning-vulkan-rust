use std::error::Error;
use std::ffi::{CStr, CString};

use ash::{Device, Entry, Instance, vk};
use ash::extensions::khr::Swapchain;
use ash::vk::{Handle, PhysicalDevice, Queue};

use winit::event_loop::EventLoop;
use winit::platform::unix::WindowExtUnix;
use winit::window::Window;
use winit::event::{Event, WindowEvent};

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
    debug: std::mem::ManuallyDrop<DebugSystem>,
    surfaces: std::mem::ManuallyDrop<SurfaceSystem>,
    physical_device: vk::PhysicalDevice,
    physical_device_properties: vk::PhysicalDeviceProperties,
    queue_families: QueueFamilies,
    queues: Queues,
    device: Device,
    swapchain: SwapchainSystem,
}

impl VulkanEngine {
    pub fn init(window: Window) -> Result<VulkanEngine, vk::Result> {
        let entry = Entry::linked();

        let layer_names = vec!["VK_LAYER_KHRONOS_validation"];

        let instance = Self::init_instance(&entry, &layer_names)?;
        let debug = DebugSystem::init(&entry, &instance)?;
        let surfaces = SurfaceSystem::init(&window, &entry, &instance)?;
        let (physical_device, physical_device_properties) = Self::init_physical_device(&instance)?;
        let queue_families = QueueFamilies::init(&instance, physical_device, &surfaces)?;
        let (device, queues) = Self::init_device_queues(&instance, physical_device, &queue_families, &layer_names)?;
        let swapchain = SwapchainSystem::init(
            &instance,
            physical_device,
            &device,
            &surfaces,
            &queue_families,
        )?;

        Ok(VulkanEngine {
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
            swapchain
        })
    }

    fn init_instance(entry: &Entry, layer_names: &[&str]) -> Result<Instance, vk::Result> {
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

    fn init_physical_device(instance: &Instance) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties), vk::Result> {
        let phys_devs = unsafe {
            instance.enumerate_physical_devices()?
        };

        let p = phys_devs[0];

        let properties = unsafe {
            instance.get_physical_device_properties(p)
        };

        Ok((p, properties))
    }

    fn init_device_queues(instance: &Instance, physical_device: PhysicalDevice, queue_families: &QueueFamilies, layer_names: &[&str]) -> Result<(Device, Queues), vk::Result> {
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
}

impl Drop for VulkanEngine{
    fn drop(&mut self) {
        unsafe {
            self.swapchain.cleanup(&self.device);
            self.device.destroy_device(None);
            std::mem::ManuallyDrop::drop(&mut self.surfaces);
            std::mem::ManuallyDrop::drop(&mut self.debug);
            self.instance.destroy_instance(None);
        }
    }
}

struct DebugSystem {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl DebugSystem {
    fn init(entry: &Entry, instance: &Instance) -> Result<DebugSystem, vk::Result> {
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

        Ok(DebugSystem {
            loader,
            messenger
        })
    }
}

impl Drop for DebugSystem {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_debug_utils_messenger(self.messenger, None)
        }
    }
}

// os specific (in my case linux)
struct SurfaceSystem {
    xlib_surface_loader: ash::extensions::khr::XlibSurface,
    surface: vk::SurfaceKHR,
    surface_loader: ash::extensions::khr::Surface,
}

impl SurfaceSystem {
    fn init(
        window: &winit::window::Window,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<SurfaceSystem, vk::Result> {
        let x11_display = window.xlib_display().unwrap();
        let x11_window = window.xlib_window().unwrap();
        let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
            .window(x11_window)
            .dpy(x11_display as *mut vk::Display);

        let xlib_surface_loader = ash::extensions::khr::XlibSurface::new(&entry, &instance);
        let surface = unsafe { xlib_surface_loader.create_xlib_surface(&x11_create_info, None)? };
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

        Ok(SurfaceSystem {
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

impl Drop for SurfaceSystem {
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
        surfaces: &SurfaceSystem,
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

struct SwapchainSystem {
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

impl SwapchainSystem {
    fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        surfaces: &SurfaceSystem,
        queue_families: &QueueFamilies,
    ) -> Result<SwapchainSystem, vk::Result> {
        let surface_capabilities = surfaces.capabilities(physical_device)?;
        let surface_present_modes = surfaces.present_modes(physical_device)?;
        let surface_formats = surfaces.formats(physical_device)?;

        let format = surface_formats.first().unwrap();

        let queue_families = [queue_families.graphics_index.unwrap()];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surfaces.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count)
            )
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(surface_capabilities.current_extent)
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

        Ok(SwapchainSystem {
            loader: swapchain_loader,
            swapchain,
            images: swapchain_images,
            image_views: swapchain_image_views
        })
    }

    unsafe fn cleanup(&mut self, device: &Device) {
        for iv in &self.image_views {
            device.destroy_image_view(*iv, None);
        }

        self.loader.destroy_swapchain(self.swapchain, None)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop)?;

    let engine = VulkanEngine::init(window)?;

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
                //drawing
            }
            _ => {}
        }
    });
}