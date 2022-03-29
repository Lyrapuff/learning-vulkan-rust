mod engine;

use ash::vk;

use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

use crate::engine::camera::Camera;
use crate::engine::model::{InstanceData, Model, TexturedInstanceData};
use crate::engine::VulkanEngine;
use crate::engine::light::{DirectionalLight, LightManager, PointLight};

use nalgebra as na;
use crate::engine::buffer::EngineBuffer;
use crate::engine::texture::Texture;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop)?;

    let mut engine = VulkanEngine::init(window)?;

    let texture = Texture::from_file("assets/Picture.png", &engine.device, &mut engine.allocator);

    let mut model = Model::quad();

    let aspect = texture.width as f32 / texture.height as f32;

    model.insert_visibly(TexturedInstanceData::from_matrix(
        na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.0))
            * na::Matrix4::new_nonuniform_scaling(&na::Vector3::new(1.0 * aspect, 1.0, 1.0))
    ));

    model.update_vertex_buffer(&mut engine.allocator).unwrap();
    model.update_index_buffer(&mut engine.allocator).unwrap();
    model.update_instance_buffer( &mut engine.allocator).unwrap();

    let models = vec![model];
    engine.models = models;

    let mut camera = Camera::builder()
        .position(na::Vector3::new(0.0, 0.0, -5.0))
        .build();

    // Maybe create an associated function for that
    let data = texture.image.clone().into_raw();

    let mut buffer = EngineBuffer::new(
        &mut engine.allocator,
        data.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::CpuToGpu,
    )?;

    buffer.fill(&mut engine.allocator, &data);
    // ^

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(engine.pools.command_pool_graphics)
        .command_buffer_count(1);

    let copy_command_buffer = unsafe {
        engine.device.allocate_command_buffers(&command_buffer_allocate_info)
    }.unwrap()[0];

    let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        engine.device.begin_command_buffer(copy_command_buffer, &cmd_begin_info)
    }?;

    let barrier = vk::ImageMemoryBarrier::builder()
        .image(texture.vk_image)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();

    unsafe {
        engine.device.cmd_pipeline_barrier(
            copy_command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    let image_subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1
    };

    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0},
        image_extent: vk::Extent3D {
            width: texture.width,
            height: texture.height,
            depth: 1
        },
        image_subresource,
        ..Default::default()
    };

    unsafe {
        engine.device.cmd_copy_buffer_to_image(
            copy_command_buffer,
            buffer.buffer,
            texture.vk_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    let barrier = vk::ImageMemoryBarrier::builder()
        .image(texture.vk_image)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();
    unsafe {
        engine.device.cmd_pipeline_barrier(
            copy_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };

    unsafe {
        engine.device.end_command_buffer(copy_command_buffer)
    }?;

    let submit_infos = [
        vk::SubmitInfo::builder()
            .command_buffers(&[copy_command_buffer])
            .build()
    ];

    let fence = unsafe {
        engine.device.create_fence(&vk::FenceCreateInfo::default(), None)
    }?;

    unsafe {
        engine.device.queue_submit(engine.queues.graphics, &submit_infos, fence)
    }?;

    unsafe {
        engine.device.wait_for_fences(&[fence], true, u64::MAX)
    }?;

    unsafe {
        engine.device.destroy_fence(fence, None)
    };

    unsafe {
        buffer.cleanup(&mut engine.allocator)
    };

    unsafe {
        engine.device.free_command_buffers(engine.pools.command_pool_graphics, &[copy_command_buffer])
    };

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = winit::event_loop::ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => match input {
                winit::event::KeyboardInput {
                    state: winit::event::ElementState::Pressed,
                    virtual_keycode: Some(keycode),
                    ..
                } => match keycode {
                    winit::event::VirtualKeyCode::Right => {
                        camera.turn_right(0.1);
                    }
                    winit::event::VirtualKeyCode::Left => {
                        camera.turn_left(0.1);
                    }
                    winit::event::VirtualKeyCode::Up => {
                        camera.move_forward(0.05);
                    }
                    winit::event::VirtualKeyCode::Down => {
                        camera.move_backward(0.05);
                    }
                    winit::event::VirtualKeyCode::PageUp => {
                        camera.turn_up(0.02);
                    }
                    winit::event::VirtualKeyCode::PageDown => {
                        camera.turn_down(0.02);
                    }
                    _ => {}
                },
                _ => {}
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

                    camera.update_buffer(&mut engine.allocator, &mut engine.uniform_buffer).unwrap();

                    for m in &mut engine.models {
                        m.update_instance_buffer( &mut engine.allocator).unwrap();
                    }

                    let image_info = vk::DescriptorImageInfo {
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        image_view: texture.image_view,
                        sampler: texture.sampler,
                        ..Default::default()
                    };

                    let descriptor_write_image = vk::WriteDescriptorSet {
                        dst_set: engine.descriptor_sets_texture[image_index as usize],
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        p_image_info: [image_info].as_ptr(),
                        ..Default::default()
                    };

                    unsafe {
                        engine.device.update_descriptor_sets(&[descriptor_write_image], &[]);
                    }

                    engine.update_command_buffer(image_index as usize)
                        .expect("Failed to update command buffer");

                    let semaphores_available = [
                        engine.swapchain.image_available[engine.swapchain.current_image]
                    ];

                    let waiting_stages = [
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    ];

                    let semaphores_finished = [
                        engine.swapchain.rendering_finished[engine.swapchain.current_image]
                    ];

                    let command_buffers = [
                        engine.graphics_command_buffers[image_index as usize]
                    ];

                    let submit_info = [
                        vk::SubmitInfo::builder()
                            .wait_semaphores(&semaphores_available)
                            .wait_dst_stage_mask(&waiting_stages)
                            .command_buffers(&command_buffers)
                            .signal_semaphores(&semaphores_finished)
                            .build()
                    ];

                    engine.device.queue_submit(
                        engine.queues.graphics,
                        &submit_info,
                        engine.swapchain.may_begin_drawing[engine.swapchain.current_image]
                    ).expect("Queue submission failed");

                    let swapchains = [engine.swapchain.swapchain];
                    let indices = [image_index];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&semaphores_finished)
                        .swapchains(&swapchains)
                        .image_indices(&indices);

                    let res = engine.swapchain.loader.queue_present(
                        engine.queues.graphics,
                        &present_info
                    );

                    match res {
                        Ok(..) => {}
                        Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                            engine.recreate_swapchain()
                                .expect("Failed to recreate swapchain");

                            camera.set_aspect(
                                engine.swapchain.extent.width as f32 /
                                    engine.swapchain.extent.height as f32
                            );

                            camera.update_buffer(&mut engine.allocator, &mut engine.uniform_buffer)
                                .expect("Failed to update Camera Uniform Buffer");
                        }
                        _ => {
                            panic!("Unhandled queue presentation error");
                        }
                    }
                }
            }
            _ => {}
        }
    });
}