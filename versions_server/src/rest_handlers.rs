use core::fmt;
use codebase::integration::deserialization::{deserialize_version};
use codebase::integration::layers_loading::load_model_xml;
use codebase::integration::serialization::{serialize_storage};
use codebase::utils::Array2F;
use warp::{Rejection, Reply, reply};
use warp::http::{StatusCode};
use warp::reply::WithStatus;
use crate::{FileManagerDep, LoadedModelDep};

type EndpointResult<T> = Result<T, Rejection>;

fn stderr_proc(e: impl fmt::Debug) -> EndpointResult<StatusCode> {
    eprintln!("{:?}", e);
    Ok(StatusCode::INTERNAL_SERVER_ERROR)
}

fn data_err_proc<T: Reply>(e: impl fmt::Debug, empty_message: T) -> EndpointResult<WithStatus<T>> {
    eprintln!("{:?}", e);
    Ok(reply::with_status(empty_message, StatusCode::INTERNAL_SERVER_ERROR))
}

pub async fn post_trainable(body: warp::hyper::body::Bytes, file_manager: FileManagerDep) -> EndpointResult<impl Reply> {
    match deserialize_version(&body) {
        Ok((storage, loss)) => {
            let mut file_manager = file_manager.write().await;
            match file_manager.add(&storage, loss) {
                Ok(_) => Ok(StatusCode::OK),
                Err(e) => return stderr_proc(e)
            }
        }
        Err(e) => return stderr_proc(e)
    }
}

pub async fn get_trainable(file_manager: FileManagerDep) -> EndpointResult<impl Reply> {
    let file_manager = file_manager.read().await;
    let most_recent = file_manager.most_recent();
    match file_manager.get_storage(most_recent) {
        Ok(storage) => {
            let bytes = serialize_storage(&storage);
            Ok(reply::with_status(bytes, StatusCode::OK))
        }
        Err(e) => {
            eprintln!("{}", e);
            Ok(reply::with_status(vec![], StatusCode::NOT_FOUND))
        }
    }
}

pub async fn get_config(file_manager: FileManagerDep) -> EndpointResult<impl Reply> {
    let file_manager = file_manager.read().await;
    match file_manager.get_config_bytes() {
        Ok(config) => Ok(reply::with_status(config, StatusCode::OK)),
        Err(_) => Ok(reply::with_status(vec![], StatusCode::NOT_FOUND)),
    }
}

pub async fn post_config(body: warp::hyper::body::Bytes, file_manager: FileManagerDep) -> EndpointResult<impl Reply> {
    match load_model_xml(&body) { // Test if the xml is valid
        Ok(_) => {
            let file_manager = file_manager.read().await;
            match file_manager.set_config_bytes(&body) {
                Ok(_) => Ok(StatusCode::OK),
                Err(e) => return stderr_proc(e)
            }
        }
        Err(_) => Ok(StatusCode::BAD_REQUEST)
    }
}

pub async fn post_eval(body: Vec<u8>, file_manager: FileManagerDep, loaded: LoadedModelDep) -> EndpointResult<impl Reply> {
    {
        // Code block to free write lock asap
        let file_manager = file_manager.read().await;
        let target = file_manager.best();
        let mut loaded = loaded.write().await;
        match loaded.assert_loaded(target, &file_manager) {
            Ok(_) => {}
            Err(e) => return data_err_proc(e, reply::json(&""))
        };
    }

    let loaded = loaded.read().await;
    let controller = loaded.get_loaded().unwrap();
    let pixels = body.into_iter().map(|o| (o as f32) / 255.0).collect();

    let inputs = match Array2F::from_shape_vec((28, 28), pixels) {
        Ok(v) => v.into_dyn(),
        Err(e) => return data_err_proc(e, reply::json(&""))
    };

    let result = match controller.eval_one(inputs) {
        Ok(v) => v,
        Err(e) => return data_err_proc(e, reply::json(&""))
    };

    let numbers: Vec<_> = result.into_iter().collect();
    Ok(reply::with_status(reply::json(&numbers), StatusCode::OK))
}