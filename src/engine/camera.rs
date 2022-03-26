use ash::Device;
use gpu_allocator::vulkan::Allocator;

use crate::engine::buffer::EngineBuffer;

use nalgebra as na;
use crate::engine::allocator::VkAllocator;

pub struct Camera {
    view_matrix: na::Matrix4<f32>,
    position: na::Vector3<f32>,
    view_direction: na::Unit<na::Vector3<f32>>,
    down_direction: na::Unit<na::Vector3<f32>>,
    fovy: f32,
    aspect: f32,
    near: f32,
    far: f32,
    projection_matrix: na::Matrix4<f32>,
}

impl Camera {
    pub fn builder() -> CameraBuilder {
        CameraBuilder {
            position: na::Vector3::new(0.0, 0.0, 0.0),
            view_direction: na::Unit::new_normalize(na::Vector3::new(0.0, 0.0, 1.0)),
            down_direction: na::Unit::new_normalize(na::Vector3::new(0.0, 1.0, 0.0)),
            fovy: std::f32::consts::FRAC_PI_3,
            aspect: 800.0 / 600.0,
            near: 0.1,
            far: 100.0,
        }
    }

    pub fn update_buffer(
        &self,
        allocator: &mut VkAllocator,
        buffer: &mut EngineBuffer
    ) -> Result<(), gpu_allocator::AllocationError> {
        let data: [[[f32; 4]; 4]; 2] = [self.view_matrix.into(), self.projection_matrix.into()];

        buffer.fill(allocator,  &data)?;

        Ok(())
    }

    pub fn update_view_matrix(&mut self) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));

        self.view_matrix = na::Matrix4::new(
            right.x,
            right.y,
            right.z,
            -right.dot(&self.position), //
            self.down_direction.x,
            self.down_direction.y,
            self.down_direction.z,
            -self.down_direction.dot(&self.position), //
            self.view_direction.x,
            self.view_direction.y,
            self.view_direction.z,
            -self.view_direction.dot(&self.position), //
            0.0,
            0.0,
            0.0,
            1.0,
        );
    }

    pub fn update_projection_matrix(&mut self) {
        let d = 1.0 / (0.5 * self.fovy).tan();

        self.projection_matrix = na::Matrix4::new(
            d / self.aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            d,
            0.0,
            0.0,
            0.0,
            0.0,
            self.far / (self.far - self.near),
            -self.near * self.far / (self.far - self.near),
            0.0,
            0.0,
            1.0,
            0.0,
        );
    }

    pub fn move_forward(&mut self, distance: f32) {
        self.position += distance * self.view_direction.as_ref();
        self.update_view_matrix();
    }

    pub fn move_backward(&mut self, distance: f32) {
        self.move_forward(-distance);
    }

    pub fn turn_right(&mut self, angle: f32) {
        let rotation = na::Rotation3::from_axis_angle(&self.down_direction, angle);
        self.view_direction = rotation * self.view_direction;
        self.update_view_matrix();
    }

    pub fn turn_left(&mut self, angle: f32) {
        self.turn_right(-angle);
    }

    pub fn turn_up(&mut self, angle: f32) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        let rotation = na::Rotation3::from_axis_angle(&right, angle);
        self.view_direction = rotation * self.view_direction;
        self.down_direction = rotation * self.down_direction;
        self.update_view_matrix();
    }

    pub fn turn_down(&mut self, angle: f32) {
        self.turn_up(-angle);
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
        self.update_projection_matrix();
    }
}

pub struct CameraBuilder {
    position: na::Vector3<f32>,
    view_direction: na::Unit<na::Vector3<f32>>,
    down_direction: na::Unit<na::Vector3<f32>>,
    fovy: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

#[allow(dead_code)]
impl CameraBuilder {
    pub fn position(mut self, pos: na::Vector3<f32>) -> CameraBuilder {
        self.position = pos;
        self
    }

    pub fn fovy(mut self, fovy: f32) -> CameraBuilder {
        self.fovy = fovy.max(0.01).min(std::f32::consts::PI - 0.01);
        self
    }

    pub fn aspect(mut self, aspect: f32) -> CameraBuilder {
        self.aspect = aspect;
        self
    }

    pub fn near(mut self, near: f32) -> CameraBuilder {
        self.near = near;
        self
    }

    pub fn far(mut self, far: f32) -> CameraBuilder {
        self.far = far;
        self
    }

    pub fn view_direction(mut self, direction: na::Vector3<f32>) -> CameraBuilder {
        self.view_direction = na::Unit::new_normalize(direction);
        self
    }

    pub fn down_direction(mut self, direction: na::Vector3<f32>) -> CameraBuilder {
        self.down_direction = na::Unit::new_normalize(direction);
        self
    }

    pub fn build(self) -> Camera {
        if self.far < self.near {
            println!(
                "far plane (at {}) closer than near plane (at {}) — is that right?",
                self.far, self.near
            );
        }

        let mut cam = Camera {
            position: self.position,
            view_direction: self.view_direction,
            down_direction: na::Unit::new_normalize(
                self.down_direction.as_ref()
                    - self
                    .down_direction
                    .as_ref()
                    .dot(self.view_direction.as_ref())
                    * self.view_direction.as_ref(),
            ),
            fovy: self.fovy,
            aspect: self.aspect,
            near: self.near,
            far: self.far,
            view_matrix: na::Matrix4::identity(),
            projection_matrix: na::Matrix4::identity(),
        };

        cam.update_projection_matrix();
        cam.update_view_matrix();

        cam
    }
}