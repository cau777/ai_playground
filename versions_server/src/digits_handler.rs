use codebase::utils::Array2F;
use warp::{reply, Reply};
use crate::{FileManagerDep, LoadedModelDep, StatusCode};
use crate::loaded_model::assert_model_loaded;
use crate::utils::{data_err_proc, EndpointResult};

pub async fn post_eval(body: Vec<u8>, file_manager: FileManagerDep, loaded: LoadedModelDep) -> EndpointResult<impl Reply> {
    {
        // Code block to free write lock asap
        let file_manager = file_manager.read().await;
        let target = file_manager.best();
        match assert_model_loaded(&loaded, target, &file_manager).await {
            Ok(_) => {}
            Err(e) => return data_err_proc(e, reply::json(&"Loading error"))
        };
    }

    let loaded = loaded.read().await;
    let controller = loaded.get_loaded().unwrap();
    let pixels = body.into_iter().map(|o| (o as f32) / 255.0).collect();

    let inputs = match Array2F::from_shape_vec((28, 28), pixels) {
        Ok(v) => v.into_dyn(),
        Err(e) => return data_err_proc(e, reply::json(&"Input error"))
    };

    let result = match controller.eval_one(inputs) {
        Ok(v) => v,
        Err(e) => return data_err_proc(e, reply::json(&"Layers error"))
    };

    let numbers: Vec<_> = result.into_iter().collect();
    Ok(reply::with_status(reply::json(&numbers), StatusCode::OK))
}