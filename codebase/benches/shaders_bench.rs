use std::sync::Arc;
use criterion::*;
use ndarray::parallel::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, DeviceExtensions, physical::PhysicalDeviceType};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{FastMemoryAllocator, MemoryUsage, StandardMemoryAllocator};
use vulkano::{VulkanLibrary, DeviceSize, sync};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;
use codebase::utils::Array4F;

fn criterion_benchmark(c: &mut Criterion) {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
            enumerate_portability: true,
            ..Default::default()
        },
    ).unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
            // that supports compute operations.
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.compute)
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        }).ok_or_else(|| "Can't find suitable device".to_owned())
        .unwrap();

    let (device, mut queues) = Device::new(physical_device, DeviceCreateInfo {
        enabled_extensions: device_extensions,
        queue_create_infos: vec![QueueCreateInfo {
            queue_family_index,
            ..Default::default()
        }],
        ..Default::default()
    }).unwrap();
    let queue = queues.next().unwrap();

    let cmd_alloc = StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_alloc = StandardDescriptorSetAllocator::new(device.clone());
    let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let data = black_box((0..200_000).map(|o| o as f32).collect::<Vec<_>>());
    let array = Array4F::from_shape_vec((2, 10, 100, 100), data.clone()).unwrap();
    let array_dyn = array.clone().into_dyn();

    let permanent_buffer_1 = CpuAccessibleBuffer::from_iter(
        &mem_alloc,
        BufferUsage {
            transfer_src: true,
            ..BufferUsage::empty()
        },
        false,
        data.iter().cloned(),
    ).unwrap();

    c.bench_function("permanent-slice",|b| b.iter(||{
        let mut write = permanent_buffer_1.write().unwrap();
        write.copy_from_slice(&data);
    }));

    c.bench_function("permanent-slice-iter",|b| b.iter(||{
        let mut write = permanent_buffer_1.write().unwrap();
        std::iter::zip(write.iter_mut(), data.iter())
            .for_each(|(a, b)| *a = *b);
    }));

    c.bench_function("permanent-array",|b| b.iter(||{
        let mut write = permanent_buffer_1.write().unwrap();
        std::iter::zip(write.iter_mut(), array.iter())
            .for_each(|(a, b)| *a = *b);
    }));

    c.bench_function("permanent-array-par",|b| b.iter(||{
        let mut write = permanent_buffer_1.write().unwrap();
        std::iter::zip(write.iter_mut(), array.iter())
            .for_each(|(a, b)| *a = *b);
    }));

    c.bench_function("permanent-array-dyn",|b| b.iter(||{
        let mut write = permanent_buffer_1.write().unwrap();
        std::iter::zip(write.iter_mut(), array_dyn.iter())
            .for_each(|(a, b)| *a = *b);
    }));

    let pool = CpuBufferPool::new(
        mem_alloc.clone(),
        BufferUsage {
            transfer_src: true,
            ..BufferUsage::empty()
        },
        MemoryUsage::Upload,
    );

    c.bench_function("pool", |b| b.iter(|| {
        pool.from_iter(
            data.iter().copied()
        ).unwrap();
    }));
    // let mut complete_data: Vec<_> = (0..1_024_000).map(|_| 0.0).collect();
    // complete_data[5_000..6_000].fill(1.0);
    //
    // c.bench_function("create set all", |b| b.iter(|| {
    //     let alloc = FastMemoryAllocator::new_default(device.clone());
    //
    //     let buffer = CpuAccessibleBuffer::from_iter(
    //         &alloc,
    //         BufferUsage {
    //             storage_buffer: true,
    //             transfer_dst: true,
    //             ..BufferUsage::empty()
    //         },
    //         false,
    //         complete_data.iter().copied(),
    //     ).unwrap();
    //
    //     dispatch(buffer, &descriptor_alloc, &cmd_alloc, device.clone(), queue.clone());
    //     cmd_alloc.clear(queue_family_index);
    //     descriptor_alloc.clear_all();
    // }));
    //
    // c.bench_function("uninit set some", |b| b.iter(|| unsafe {
    //     let alloc = FastMemoryAllocator::new_default(device.clone());
    //
    //     let buffer = CpuAccessibleBuffer::<[f32]>::uninitialized_array(
    //         &alloc,
    //         complete_data.len() as DeviceSize,
    //         BufferUsage {
    //             storage_buffer: true,
    //             transfer_dst: true,
    //             ..BufferUsage::empty()
    //         },
    //         false,
    //     ).unwrap();
    //
    //     {
    //         let mut write = buffer.write().unwrap();
    //         let mut write = write.as_mut();
    //         write[5_000..6_000].copy_from_slice(&complete_data[5_000..6_000]);
    //     }
    //
    //     dispatch(buffer, &descriptor_alloc, &cmd_alloc, device.clone(), queue.clone());
    //
    //     cmd_alloc.clear(queue_family_index);
    //     descriptor_alloc.clear_all();
    // }));
    //
    // let std_alloc=StandardMemoryAllocator::new_default(device.clone());
    // let buffer = CpuAccessibleBuffer::from_iter(
    //     &std_alloc,
    //     BufferUsage {
    //         storage_buffer: true,
    //         transfer_dst: true,
    //         ..BufferUsage::empty()
    //     },
    //     false,
    //     complete_data.iter().copied(),
    // ).unwrap();
    // c.bench_function("static buffer", |b| b.iter(|| {
    //     {
    //         let mut write = buffer.write().unwrap();
    //         let mut write = write.as_mut();
    //         write[5_000..6_000].copy_from_slice(&complete_data[5_000..6_000]);
    //     }
    //
    //     dispatch(buffer.clone(), &descriptor_alloc, &cmd_alloc, device.clone(), queue.clone());
    //
    //     cmd_alloc.clear(queue_family_index);
    //     descriptor_alloc.clear_all();
    // }));
}

fn dispatch(buffer: Arc<CpuAccessibleBuffer<[f32]>>, descriptor_alloc: &StandardDescriptorSetAllocator,
            cmd_alloc: &StandardCommandBufferAllocator, device: Arc<Device>, queue: Arc<Queue>) {
    let pipeline = {
        let shader = cs::load(device.clone()).unwrap();
        ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        ).unwrap()
    };

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        descriptor_alloc,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, buffer.clone())],
    ).unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        cmd_alloc,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([250, 1, 1])
        .unwrap();
    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
            #version 450
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            layout(set = 0, binding = 0) buffer Data {
                float data[];
            } data;
            void main() {
                float sum = 0.0;
                for (uint x = 0; x < 64; x++) { sum += data.data[gl_GlobalInvocationID.x * 64 + x]; }
            }
        "
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);