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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop)?;

    let mut engine = VulkanEngine::init(window)?;

    let mut model = Model::quad();

    model.insert_visibly(TexturedInstanceData::from_matrix(
        na::Matrix4::new_translation(&na::Vector3::new(0.0, 0.0, 0.0))
            * na::Matrix4::new_scaling(0.5)
    ));

    model.update_vertex_buffer(&mut engine.allocator).unwrap();
    model.update_index_buffer(&mut engine.allocator).unwrap();
    model.update_instance_buffer( &mut engine.allocator).unwrap();

    let models = vec![model];
    engine.models = models;

    /*

    let mut lights = LightManager::default();

    lights.add_light(DirectionalLight {
        direction: na::Vector3::new(-1., -1., 0.),
        illuminance: [10.1, 10.1, 10.1],
    });
    lights.add_light(PointLight {
        position: na::Point3::new(0.1, -3.0, -3.0),
        luminous_flux: [100.0, 100.0, 100.0],
    });
    lights.add_light(PointLight {
        position: na::Point3::new(0.1, -3.0, -3.0),
        luminous_flux: [100.0, 100.0, 100.0],
    });
    lights.add_light(PointLight {
        position: na::Point3::new(0.1, -3.0, -3.0),
        luminous_flux: [100.0, 100.0, 100.0],
    });

    lights.update_buffer(&engine.device, &mut engine.allocator, &mut engine.light_buffer, &mut engine.descriptor_sets_light).unwrap();

    */

    let mut camera = Camera::builder()
        .position(na::Vector3::new(0.0, 0.0, -5.0))
        .build();

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

                    engine.device.queue_submit(
                        engine.queues.graphics,
                        &submit_info,
                        engine.swapchain.may_begin_drawing[engine.swapchain.current_image]
                    ).expect("Queue submission");

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