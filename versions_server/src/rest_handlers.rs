use codebase::integration::deserialization::{deserialize_version};
use codebase::integration::layers_loading::load_model_xml;
use codebase::integration::serialization::{serialize_storage};
use warp::{Reply, reply};
use warp::http::{StatusCode};
use crate::{FileManagersDep};
use crate::utils::{EndpointResult, stderr_proc};

/// Returns the trainable parameters of a specific model
pub async fn get_trainable(name: String, file_managers: FileManagersDep) -> EndpointResult<impl Reply> {
    let file_manager = match file_managers.get_from_name(&name) {
        Some(v) => v,
        None => return Ok(reply::with_status(vec![], StatusCode::NOT_FOUND)),
    };

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

/// Adds the trainable parameters supplied by the user as a new version of the model
pub async fn post_trainable(name: String, body: warp::hyper::body::Bytes, file_managers: FileManagersDep) -> EndpointResult<impl Reply> {
    let file_manager = match file_managers.get_from_name(&name) {
        Some(v) => v,
        None => return Ok(StatusCode::NOT_FOUND)
    };

    match deserialize_version(&body) {
        Ok((storage, loss)) => {
            let mut file_manager = file_manager.write().await;
            match file_manager.add(&storage, loss) {
                Ok(_) => Ok(StatusCode::OK),
                Err(e) => stderr_proc(e)
            }
        }
        Err(e) => stderr_proc(e)
    }
}

/// Returns the config for a specific model
pub async fn get_config(name: String, file_managers: FileManagersDep) -> EndpointResult<impl Reply> {
    let file_manager = match file_managers.get_from_name(&name) {
        Some(v) => v,
        None => return Ok(reply::with_status(vec![], StatusCode::NOT_FOUND))
    };

    let file_manager = file_manager.read().await;
    match file_manager.get_config_bytes() {
        Ok(config) => Ok(reply::with_status(config, StatusCode::OK)),
        Err(_) => Ok(reply::with_status(vec![], StatusCode::NOT_FOUND)),
    }
}

/// Replaces the config for a specific model with the one provided by the user, if it's valid
pub async fn post_config(name: String, body: warp::hyper::body::Bytes, file_managers: FileManagersDep) -> EndpointResult<impl Reply> {
    let file_manager = match file_managers.get_from_name(&name) {
        Some(v) => v,
        None => return Ok(StatusCode::NOT_FOUND)
    };

    match load_model_xml(&body) { // Test if the xml is valid
        Ok(_) => {
            let file_manager = file_manager.read().await;
            match file_manager.set_config_bytes(&body) {
                Ok(_) => Ok(StatusCode::OK),
                Err(e) => stderr_proc(e),
            }
        }
        Err(_) => Ok(StatusCode::BAD_REQUEST)
    }
}
