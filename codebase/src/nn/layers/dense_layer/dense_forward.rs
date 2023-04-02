use crate::ArrayDynF;
use crate::gpu::gpu_data::GlobalGpu;
use crate::gpu::shader_context::{BufferConfig, ContextBinding, ShaderBinding, ShaderContext};
use crate::gpu::shader_runner_2::ShaderRunner2;
use crate::nn::layers::dense_layer::*;
use crate::nn::layers::nn_layers::*;
use crate::nn::layers::stored_array::StoredArray;
use crate::utils::{Array1F, Array2F, GenericResult, shape_length};
use crate::gpu::{BufferChecksumMethod, shaders};

/// Perform matrix multiplication between the input and a weights matrix, and add a biases matrix.
pub fn forward(data: ForwardData, layer_config: &DenseConfig) -> LayerResult {
    let ForwardData {
        assigner,
        storage,
        inputs,
        forward_cache,
        gpu,
        ..
    } = data;
    let ish = inputs.shape();
    if ish.len() != 2 {
        return Err(anyhow::anyhow!("Inputs should be bidimensional"));
    }
    if ish[1] != layer_config.in_values {
        return Err(anyhow::anyhow!("Input length {} does not match the one specified in the config {}", ish[1], layer_config.in_values));
    }

    let key = assigner.get_key(gen_name(layer_config));
    if let Some(forward_cache) = forward_cache {
        forward_cache.insert(key.clone(), vec![inputs.to_memory()?]);
    }

    let [weights, biases] = get_from_storage2(storage, &key);

    let result = if matches!(inputs, StoredArray::GpuLocal {..}) {
        match forward_gpu(key, &inputs, weights, biases, gpu.unwrap(), layer_config) {
            Ok(v) => v,
            Err(_e) => {
                #[cfg(debug_assertions)]
                println!("{}", _e);
                forward_cpu(inputs, weights, biases)?
            }
        }
    } else {
        forward_cpu(inputs, weights, biases)?
    };

    Ok(result)
}

fn forward_cpu(inputs: StoredArray, weights: &ArrayDynF, biases: &ArrayDynF) -> GenericResult<StoredArray> {
    let weights: Array2F = weights.clone().into_dimensionality()?;
    let biases: &Array1F = &biases.clone().into_dimensionality()?;
    let inputs: Array2F = inputs.into_memory()?.into_dimensionality()?;

    let mut dot_prod = Vec::with_capacity(inputs.batch_size());
    inputs
        .outer_iter()
        .into_par_iter()
        .map(|o| weights.dot(&o))
        .map(|o| o + biases)
        .collect_into_vec(&mut dot_prod);

    let mut views = Vec::with_capacity(inputs.batch_size());
    views.extend(dot_prod.iter().map(|o| o.view()));

    let result = stack(Axis(0), &views)?;
    Ok(result.into_dyn().into())
}

fn forward_gpu(key: String, inputs: &StoredArray, weights: &ArrayDynF, biases: &ArrayDynF,
               gpu: GlobalGpu, layer_config: &DenseConfig) -> GenericResult<StoredArray> {
    let id = (key, "forward".to_owned());
    let ish = inputs.shape();
    let buffers = [
        BufferConfig::floats(ish[0] * layer_config.out_values),
        BufferConfig::floats(layer_config.out_values * layer_config.in_values),
        BufferConfig::floats(layer_config.out_values),
        BufferConfig::floats(shape_length(ish)),
    ];
    ShaderContext::register(&id, gpu.clone(), &buffers, |mut b| {
        b.register_shader("forward", shaders::dense_forward::load, vec![
            (ContextBinding(0), ShaderBinding(0)),
            (ContextBinding(1), ShaderBinding(1)),
            (ContextBinding(2), ShaderBinding(2)),
            (ContextBinding(3), ShaderBinding(3)),
        ], &shaders::dense_forward::SpecializationConstants {
            in_values: layer_config.in_values as u32,
            out_values: layer_config.out_values as u32,
        })?;
        Ok(b)
    })?;

    let mut runner = ShaderRunner2::new(id, gpu.clone())?;
    runner.update_buffer_with_memory(ContextBinding(1), weights, BufferChecksumMethod::Single)?
        .update_buffer_with_memory(ContextBinding(2), biases, BufferChecksumMethod::Single)?
        .update_buffer_with_stored_array(ContextBinding(3), inputs, BufferChecksumMethod::Split)?
        .dispatch("forward", [ish[0], layer_config.out_values, 1].map(|o| o as u32), shaders::dense_forward::BLOCK_SIZE)?;

    let result = runner.finish()?;

    Ok(StoredArray::GpuLocal { data: result, gpu, shape: vec![ish[0], layer_config.out_values] })
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use crate::gpu::buffers::upload_array_to_gpu;
    use crate::gpu::gpu_data::get_global_gpu;
    use crate::utils::Array2F;
    use crate::nn::batch_config::BatchConfig;
    use crate::nn::key_assigner::KeyAssigner;
    use crate::nn::layers::dense_layer::tests::get_config;
    use crate::nn::lr_calculators::constant_lr::ConstantLrConfig;
    use crate::utils::arrays_almost_equal;
    use super::*;

    #[test]
    fn test_forward() {
        let input = array![[1.0, 2.0], [2.0, 3.0]].into_dyn();
        let weights = array![[0.7, 0.0], [0.1, 0.4], [0.8, 0.6]];
        let expected = array![[0.7, 0.9, 2.0], [1.4, 1.4, 3.4]].into_dyn();

        let config = get_config(DenseLayerInit::WeightsAndBiases(weights, Array1F::zeros(3)));

        let mut storage = GenericStorage::new();
        DenseLayer::init(
            InitData {
                storage: &mut storage,
                assigner: &mut KeyAssigner::new(),
            },
            &config,
        )
            .unwrap();

        let output = forward(
            ForwardData {
                batch_config: &BatchConfig::new_train(),
                assigner: &mut KeyAssigner::new(),
                storage: &mut storage,
                inputs: input.into(),
                forward_cache: None,
                prev_iteration_cache: None,
                gpu: None,
            },
            &config,
        ).unwrap();

        assert!(arrays_almost_equal(&output.into_memory().unwrap(), &expected));
    }

    #[test]
    fn test_cpu_equals_gpu() {
        let dist = Normal::new(0.0, 1.0).unwrap();
        let config = DenseConfig {
            in_values: 64,
            out_values: 32,
            weights_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            biases_lr_calc: LrCalc::Constant(ConstantLrConfig::default()),
            init_mode: DenseLayerInit::Random(),
        };
        let gpu = get_global_gpu().unwrap();

        let inputs = Array2F::random((8, config.in_values), &dist).into_dyn();
        let weights = Array2F::random((config.out_values, config.in_values), &dist).into_dyn();
        let biases = Array1F::random((config.out_values), &dist).into_dyn();

        let expected = forward_cpu(inputs.clone().into(), &weights, &biases)
            .unwrap().into_memory().unwrap();

        let data = upload_array_to_gpu(&inputs, &gpu).unwrap();
        let actual = forward_gpu("test_cpu_equals_gpu".to_owned(),
                                 &StoredArray::GpuLocal { data, gpu: gpu.clone(), shape: inputs.shape().to_vec() },
                                 &weights,
                                 &biases,
                                 gpu,
                                 &config,
        ).unwrap().into_memory().unwrap();

        println!("{:?}\n------------\n{:?}", actual, expected);
        assert!(arrays_almost_equal(&actual, &expected));
    }
}