use std::sync::Arc;
use std::time::Instant;

use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::device::physical::PhysicalDevice;
use vulkano::format::Format;
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};

use vulkano_win::VkSurfaceBuild;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use nalgebra_glm::{identity, look_at, perspective, pi, rotate_normalized_axis, TMat4, translate, vec3};

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal, color);

#[derive(Debug, Clone)]
struct MVP {
    model: TMat4<f32>,
    view: TMat4<f32>,
    projection: TMat4<f32>
}

impl MVP {
    fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity()
        }
    }
}

#[derive(Default, Debug, Clone)]
struct AmbientLight {
    color: [f32; 3],
    intensity: f32
}

#[derive(Default, Debug, Clone)]
struct DirectionalLight {
    position: [f32; 4],
    color: [f32; 3]
}

fn main() {
    let mut mvp = MVP::new();
    mvp.view = look_at(&vec3(0.0, 0.0, 0.01), &vec3(0.0, 0.0, 0.0), &vec3(0.0, -1.0, 0.0));
    mvp.model = translate(&identity(), &vec3(0.0, 0.0, -2.5));

    let ambient_light = AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.2
    };

    let directional_light = DirectionalLight {
        position: [-4.0, -4.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0]
    };

    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, Version::V1_1, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    let queue_family = physical.queue_families().find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false)).unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (device, mut queues) = Device::new(
        physical,
        &Features::none(),
        &physical.required_extensions().union(&device_ext),
        [(queue_family, 0.5)].iter().cloned(),
    ).unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(device.physical_device()).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        mvp.projection = perspective(
            dimensions[0] as f32 / dimensions[1] as f32,
            180.0,
            0.01,
            100.0
        );

        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(usage)
            .sharing_mode(&queue)
            .composite_alpha(alpha)
            .build()
            .unwrap()
    };

    mod deferred_vert {
        vulkano_shaders::shader!{
            ty: "vertex",
            path: "src/shaders/deferred.vert"
        }
    }

    mod deferred_frag {
        vulkano_shaders::shader!{
            ty: "fragment",
            path: "src/shaders/deferred.frag"
        }
    }

    mod lighting_vert {
        vulkano_shaders::shader!{
            ty: "vertex",
            path: "src/shaders/lighting.vert"
        }
    }

    mod lighting_frag {
        vulkano_shaders::shader!{
            ty: "fragment",
            path: "src/shaders/lighting.frag"
        }
    }

    let deferred_vert = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag::load(device.clone()).unwrap();
    let lighting_vert = lighting_vert::load(device.clone()).unwrap();
    let lighting_frag = lighting_frag::load(device.clone()).unwrap();

    let uniform_buffer = CpuBufferPool::<deferred_vert::ty::MVP_Data>::uniform_buffer(device.clone());
    let ambient_buffer = CpuBufferPool::<lighting_frag::ty::Ambient_Data>::uniform_buffer(device.clone());
    let directional_buffer = CpuBufferPool::<lighting_frag::ty::Directional_Light_Data>::uniform_buffer(device.clone());

    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            color: {
                load: Clear,
                store: DontCare,
                format: Format::A2B10G10R10_UNORM_PACK32,
                samples: 1,
            },
            normals: {
                load: Clear,
                store: DontCare,
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        passes: [
            {
                color: [color, normals],
                depth_stencil: {depth},
                input: []
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color, normals]
            }
        ]
    ).unwrap();

    let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
    let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

    let deferred_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(deferred_pass)
        .build(device.clone())
        .unwrap();

    let lighting_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(lighting_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(lighting_frag.entry_point("main").unwrap(), ())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_pass)
        .build(device.clone())
        .unwrap();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
        // front face
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},

        // back face
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},

        // top face
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},

        // bottom face
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},

        // left face
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},

        // right face
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
    ].iter().cloned()
    ).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let (mut framebuffers, mut color_buffer, mut normal_buffer) =
        window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let rotation_start = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            },
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().take().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    mvp.projection = perspective(
                        dimensions[0] as f32 / dimensions[1] as f32,
                        180.0,
                        0.01,
                        100.0
                    );

                    let (new_swapchain, new_images) = match swapchain.recreate().dimensions(dimensions).build() {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };

                    swapchain = new_swapchain;

                    let (new_framebuffers, new_color_buffer, new_normal_buffer) =
                        window_size_dependent_setup(device.clone(), &new_images, render_pass.clone(), &mut viewport);

                    framebuffers = new_framebuffers;
                    color_buffer = new_color_buffer;
                    normal_buffer = new_normal_buffer;

                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed().as_secs() as f64 + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                    let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;

                    let mut model:TMat4<f32> = rotate_normalized_axis(&identity(), elapsed_as_radians as f32 * 50.0, &vec3(0.0, 0.0, 1.0));
                    model = rotate_normalized_axis(&model, elapsed_as_radians as f32 * 30.0, &vec3(0.0, 1.0, 0.0));
                    model = rotate_normalized_axis(&model, elapsed_as_radians as f32 * 20.0, &vec3(1.0, 0.0, 0.0));
                    model = mvp.model * model;

                    let uniform_data = deferred_vert::ty::MVP_Data {
                        model: model.into(),
                        view: mvp.view.into(),
                        projection: mvp.projection.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let ambient_uniform_subbuffer = {
                    let uniform_data = lighting_frag::ty::Ambient_Data {
                        color: ambient_light.color.into(),
                        intensity: ambient_light.intensity.into()
                    };

                    ambient_buffer.next(uniform_data).unwrap()
                };

                let directional_uniform_subbuffer = {
                    let uniform_data = lighting_frag::ty::Directional_Light_Data {
                        position: directional_light.position.into(),
                        color: directional_light.color.into()
                    };

                    directional_buffer.next(uniform_data).unwrap()
                };

                let deferred_layout = deferred_pipeline
                    .layout()
                    .descriptor_set_layouts()
                    .get(0)
                    .unwrap();
                let mut deferred_set_builder = PersistentDescriptorSet::start(deferred_layout.clone());
                deferred_set_builder
                    .add_buffer(uniform_buffer_subbuffer.clone())
                    .unwrap();
                let deferred_set = deferred_set_builder.build().unwrap();

                let lighting_layout = lighting_pipeline
                    .layout()
                    .descriptor_set_layouts()
                    .get(0)
                    .unwrap();
                let mut lighting_set_builder = PersistentDescriptorSet::start(lighting_layout.clone());
                lighting_set_builder
                    .add_image(color_buffer.clone())
                    .unwrap();
                lighting_set_builder
                    .add_image(normal_buffer.clone())
                    .unwrap();
                lighting_set_builder
                    .add_buffer(uniform_buffer_subbuffer)
                    .unwrap();
                lighting_set_builder
                    .add_buffer(ambient_uniform_subbuffer)
                    .unwrap();
                lighting_set_builder
                    .add_buffer(directional_uniform_subbuffer)
                    .unwrap();
                let lighting_set = lighting_set_builder.build().unwrap();

                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), [0.0, 0.0, 0.0, 1.0].into(), [0.0, 0.0, 0.0, 1.0].into(), 1f32.into()];

                let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit
                ).unwrap();

                cmd_buffer_builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        clear_values
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(deferred_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        deferred_pipeline.layout().clone(),
                        0,
                        deferred_set.clone()
                    )
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .next_subpass(SubpassContents::Inline)
                    .unwrap()
                    .bind_pipeline_graphics(lighting_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        lighting_pipeline.layout().clone(),
                        0,
                        lighting_set.clone()
                    )
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = cmd_buffer_builder.build().unwrap();

                let future = previous_frame_end.take().unwrap().join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }
            },
            _ => {}
        }
    });
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> (
    Vec<Arc<Framebuffer>>,
    Arc<ImageView<AttachmentImage>>,
    Arc<ImageView<AttachmentImage>>,
) {
    let dimensions = images[0].dimensions().width_height();

    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let depth_buffer = ImageView::new(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    ).unwrap();

    let color_buffer = ImageView::new(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::A2B10G10R10_UNORM_PACK32,
        ).unwrap()
    ).unwrap();

    let normal_buffer = ImageView::new(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::R16G16B16A16_SFLOAT
        ).unwrap()
    ).unwrap();

    (
        images
            .iter()
            .map(|image| {
                let view = ImageView::new(image.clone()).unwrap();

                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .add(color_buffer.clone())
                    .unwrap()
                    .add(normal_buffer.clone())
                    .unwrap()
                    .add(depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>(),
        color_buffer.clone(),
        normal_buffer.clone()
    )
}