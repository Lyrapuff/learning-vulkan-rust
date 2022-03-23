use std::collections::HashMap;
use gpu_allocator::vulkan::Allocator;
use super::buffer::EngineBuffer;
use ash::vk;

#[derive(Debug, Clone)]
pub struct InvalidHandle;

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
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub color: [f32; 3],
}

pub struct Model<V, I> {
    pub vertex_data: Vec<V>,
    pub index_data: Vec<u32>,
    pub handle_to_index: HashMap<usize, usize>,
    pub handles: Vec<usize>,
    pub instances: Vec<I>,
    pub first_invisible: usize,
    pub next_handle: usize,
    pub vertex_buffer: Option<EngineBuffer>,
    pub index_buffer: Option<EngineBuffer>,
    pub instance_buffer: Option<EngineBuffer>,
}

impl<V, I> Model<V, I> {
    pub fn get(&self, handle: usize) -> Option<&I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get(index)
        }
        else {
            None
        }
    }

    pub fn get_mut(&mut self, handle: usize) -> Option<&mut I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get_mut(index)
        } else {
            None
        }
    }

    pub fn swap_by_handle(&mut self, h1: usize, h2: usize) -> Result<(), InvalidHandle> {
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

    pub fn swap_by_index(&mut self, index1: usize, index2: usize) {
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

    pub fn is_visible(&self, handle: usize) -> Result<bool, InvalidHandle> {
        if let Some(index) = self.handle_to_index.get(&handle) {
            Ok(index < &self.first_invisible)
        }
        else {
            Err(InvalidHandle)
        }
    }

    pub fn make_visible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
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

    pub fn make_invisible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
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

    pub fn insert(&mut self, element: I) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;

        let index = self.instances.len();
        self.instances.push(element);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);

        handle
    }

    pub fn insert_visibly(&mut self, element: I) -> usize {
        let new_handle = self.insert(element);
        self.make_visible(new_handle).ok();
        new_handle
    }

    pub fn remove(&mut self, handle: usize) -> Result<I, InvalidHandle> {
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

    pub fn update_vertex_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator
    ) -> Result<(), gpu_allocator::AllocationError> {
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

    pub fn update_index_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator
    ) -> Result<(), gpu_allocator::AllocationError> {
        if let Some(buffer) = &mut self.index_buffer {
            buffer.fill(allocator, device, &self.index_data)?;
            Ok(())
        } else {
            let bytes = (self.index_data.len() * std::mem::size_of::<u32>()) as u64;
            let mut buffer = EngineBuffer::new(
                allocator,
                device,
                bytes,
                vk::BufferUsageFlags::INDEX_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;

            buffer.fill(allocator, device, &self.index_data)?;
            self.index_buffer = Some(buffer);

            Ok(())
        }
    }

    pub fn update_instance_buffer(&mut self, device: &ash::Device, allocator: &mut Allocator) -> Result<(), gpu_allocator::AllocationError> {
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

    pub fn draw(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if let Some(index_buffer) = &self.index_buffer {
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

                            device.cmd_bind_index_buffer(
                                command_buffer,
                                index_buffer.buffer,
                                0,
                                vk::IndexType::UINT32,
                            );

                            device.cmd_draw_indexed(
                                command_buffer,
                                self.index_data.len() as u32,
                                self.first_invisible as u32,
                                0,
                                0,
                                0,
                            );
                        }
                    }
                }
            }
        }
    }
}

impl Model<[f32; 3], InstanceData> {
    pub fn cube() -> Self {
        let lbf = [-1.0,1.0,0.0]; //lbf: left-bottom-front
        let lbb = [-1.0,1.0,1.0];
        let ltf = [-1.0,-1.0,0.0];
        let ltb = [-1.0,-1.0,1.0];
        let rbf = [1.0,1.0,0.0];
        let rbb = [1.0,1.0,1.0];
        let rtf = [1.0,-1.0,0.0];
        let rtb = [1.0,-1.0,1.0];

        Model {
            vertex_data: vec![lbf, lbb, ltf, ltb, rbf, rbb, rtf, rtb],
            index_data: vec![
                0, 1, 5, 0, 5, 4, //bottom
                2, 7, 3, 2, 6, 7, //top
                0, 6, 2, 0, 4, 6, //front
                1, 3, 7, 1, 7, 5, //back
                0, 2, 1, 1, 2, 3, //left
                4, 5, 6, 5, 7, 6, //right
            ],
            handle_to_index: HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }
}