pub mod buffer;
pub mod debug;
pub mod surface;
pub mod queue_families;
pub mod swapchain;
pub mod pipeline;
pub mod pools;
pub mod model;

pub mod camera;
pub mod light;

use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;

use ash::{Device, Entry, Instance, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use winit::window::Window;

use nalgebra as na;

use crate::engine::buffer::EngineBuffer;
use crate::engine::debug::EngineDebug;
use crate::engine::model::{InstanceData, Model, VertexData};
use crate::engine::pipeline::EnginePipeline;
use crate::engine::pools::Pools;
use crate::engine::queue_families::QueueFamilies;
use crate::engine::surface::EngineSurface;
use crate::engine::swapchain::EngineSwapchain;

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

pub struct VulkanEngine {
    pub window: Window,
    pub entry: Entry,
    pub instance: Instance,
    pub debug: ManuallyDrop<EngineDebug>,
    pub surfaces: ManuallyDrop<EngineSurface>,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub queue_families: QueueFamilies,
    pub queues: Queues,
    pub device: Device,
    pub swapchain: EngineSwapchain,
    pub render_pass: vk::RenderPass,
    pub pipeline: EnginePipeline,
    pub pools: Pools,
    pub graphics_command_buffers: Vec<vk::CommandBuffer>,
    pub allocator: ManuallyDrop<Allocator>,
    pub models: Vec<Model<VertexData, InstanceData>>,
    pub uniform_buffer: EngineBuffer,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets_cam: Vec<vk::DescriptorSet>,
    pub descriptor_sets_light: Vec<vk::DescriptorSet>,
    pub light_buffer: EngineBuffer,
}

impl VulkanEngine {
    pub fn init(window: Window) -> Result<VulkanEngine, vk::Result> {
        let entry = Entry::linked();

        let layer_names = vec!["VK_LAYER_KHRONOS_validation"];

        let instance = Self::init_instance(&entry, &layer_names)?;

        let debug = EngineDebug::init(&entry, &instance, Some(vulkan_debug_utils_callback))?;

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

        let mut uniform_buffer = EngineBuffer::new(
            &mut allocator,
            &device,
            128,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu
        ).unwrap();

        let camera_transforms: [[[f32; 4]; 4]; 2] = [
            na::Matrix4::identity().into(),
            na::Matrix4::identity().into(),
        ];

        uniform_buffer.fill(&mut allocator, &device, &camera_transforms).unwrap();

        // Descriptor pool

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain.amount_of_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: swapchain.amount_of_images,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(2 * swapchain.amount_of_images) //
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }?;

        let desc_layouts_camera =
            vec![pipeline.descriptor_set_layouts[0]; swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info_camera = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts_camera);
        let descriptor_sets_camera = unsafe {
            device.allocate_descriptor_sets(&descriptor_set_allocate_info_camera)
        }?;

        for descset in &descriptor_sets_camera {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: 128,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { device.update_descriptor_sets(&desc_sets_write, &[]) };
        }
        let desc_layouts_light =
            vec![pipeline.descriptor_set_layouts[1]; swapchain.amount_of_images as usize];
        let descriptor_set_allocate_info_light = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts_light);
        let descriptor_sets_light = unsafe {
            device.allocate_descriptor_sets(&descriptor_set_allocate_info_light)
        }?;

        let mut light_buffer = EngineBuffer::new(
            &mut allocator,
            &device,
            8,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
        ).unwrap();

        light_buffer.fill(&mut allocator, &device, &[0., 0.]).unwrap();

        for descset in &descriptor_sets_light {
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: light_buffer.buffer,
                offset: 0,
                range: 8,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*descset)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { device.update_descriptor_sets(&desc_sets_write, &[]) };
        }

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
            uniform_buffer,
            descriptor_pool,
            descriptor_sets_cam: descriptor_sets_camera,
            descriptor_sets_light: descriptor_sets_light,
            light_buffer,
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
        physical_device: vk::PhysicalDevice,
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
        physical_device: vk::PhysicalDevice,
        surfaces: &EngineSurface
    ) -> Result<vk::RenderPass, vk::Result> {
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

    pub fn update_command_buffer(&mut self, index: usize) -> Result<(), vk::Result> {
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

            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[
                    self.descriptor_sets_cam[index],
                    self.descriptor_sets_light[index]
                ],
                &[],
            );

            for m in &self.models {
                m.draw(&self.device, command_buffer);
            }

            self.device.cmd_end_render_pass(command_buffer);
            self.device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    fn fill_command_buffers(&self, models: &[Model<VertexData, InstanceData>]) {
        for (i, &command_buffer) in self.graphics_command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();

            unsafe {
                self.device.begin_command_buffer(command_buffer, &command_buffer_begin_info).unwrap();
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

                self.device.end_command_buffer(command_buffer).unwrap();
            }
        }
    }
}

impl Drop for VulkanEngine{
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("Failed to wait?");

            self.light_buffer.cleanup(&mut self.allocator, &self.device);

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);

            self.uniform_buffer.cleanup(&mut self.allocator, &self.device);

            for m in &mut self.models {
                if let Some(vb) = &mut m.vertex_buffer {
                    vb.cleanup(&mut self.allocator, &self.device);
                }

                if let Some(ib) = &mut m.index_buffer {
                    ib.cleanup(&mut self.allocator, &self.device);
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

pub struct Queues {
    pub graphics: vk::Queue,
    pub transfer: vk::Queue,
}