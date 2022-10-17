use crate::compiled_protos::common::{ModelStorageData, NdArrayData, PairData};
use crate::compiled_protos::model_storage::ArrayPairData;
use crate::compiled_protos::task_result::{TaskResultData, TestTaskData};
use crate::integration::array_pair::ArrayPair;
use crate::nn::layers::nn_layers::GenericStorage;
use crate::utils::ArrayDynF;
use protobuf::{Message, MessageField};

pub enum TaskResult {
    Train(GenericStorage),
    Test(u32, u32, f64),
}

pub fn load_model_from_bytes(bytes: &[u8]) -> Option<GenericStorage> {
    let data = ModelStorageData::parse_from_bytes(bytes).ok()?;
    load_model(data)
}

fn load_model(data: ModelStorageData) -> Option<GenericStorage> {
    let mut result = GenericStorage::new();

    for pair in data.pairs.into_iter() {
        let mut arrays = Vec::new();

        for arr in pair.arrays.into_iter() {
            arrays.push(load_array(arr)?);
        }

        result.insert(pair.key, arrays);
    }

    Some(result)
}

pub fn load_pair_from_bytes(bytes: &[u8]) -> Option<ArrayPair> {
    let data = ArrayPairData::parse_from_bytes(bytes).ok()?;
    let ArrayPairData {
        expected, inputs, ..
    } = data;

    Some(ArrayPair {
        inputs: load_array(inputs.into_option()?)?,
        expected: load_array(expected.into_option()?)?,
    })
}

fn load_array(data: NdArrayData) -> Option<ArrayDynF> {
    let shape: Vec<usize> = data.shape.iter().map(|o| *o as usize).collect();
    ArrayDynF::from_shape_vec(shape, data.numbers).ok()
}

pub fn load_task_result_from_bytes(bytes: &[u8]) -> Option<TaskResult> {
    use crate::compiled_protos::task_result::task_result_data::Result::*;

    let data = TaskResultData::parse_from_bytes(bytes).ok()?;
    match data.result.unwrap() {
        Delta(delta) => Some(TaskResult::Train(load_model(delta)?)),
        TestData(data) => Some(TaskResult::Test(data.version, data.batch, data.accuracy)),
    }
}

pub fn save_model_bytes(model: &GenericStorage) -> protobuf::Result<Vec<u8>> {
    save_model(model).write_to_bytes()
}

fn save_model(model: &GenericStorage) -> ModelStorageData {
    let mut data = ModelStorageData::new();

    for (key, value) in model.iter() {
        let arrays = value.iter().map(save_array).collect();

        let mut pair = PairData::new();
        pair.key = key.clone();
        pair.arrays = arrays;
        data.pairs.push(pair);
    }

    data
}

pub fn save_array_pair_bytes(array: &ArrayPair) -> Option<Vec<u8>> {
    let mut data = ArrayPairData::new();
    data.inputs = MessageField::some(save_array(&array.inputs));
    data.expected = MessageField::some(save_array(&array.expected));
    data.write_to_bytes().ok()
}

fn save_array(array: &ArrayDynF) -> NdArrayData {
    let mut data = NdArrayData::new();
    data.shape = array.shape().iter().map(|o| *o as u32).collect();
    data.numbers = array.iter().copied().collect();
    data
}

pub fn save_task_result_bytes(result: TaskResult) -> protobuf::Result<Vec<u8>> {
    use TaskResult::*;
    let mut data = TaskResultData::new();

    match result {
        Train(deltas) => {
            data.set_delta(save_model(&deltas));
        }
        Test(version, batch,  accuracy) => {
            let mut test_data = TestTaskData::new();
            test_data.version = version;
            test_data.accuracy = accuracy;
            test_data.batch = batch;
            data.set_test_data(test_data);
        }
    }

    data.write_to_bytes()
}
