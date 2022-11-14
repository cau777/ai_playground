use std::str::FromStr;
use codebase::integration::deserialization::deserialize_version;
use codebase::integration::layers_loading::load_model_xml;
use codebase::integration::serialization::serialize_storage;
use warp::{Rejection, Reply, reply};
use warp::http::{StatusCode, Uri};
use crate::{EnvConfigDep, FileManagerDep};

type EndpointResult<T> = Result<T, Rejection>;

pub async fn post_trainable(body: warp::hyper::body::Bytes, file_manager: FileManagerDep) -> EndpointResult<impl Reply> {
    match deserialize_version(&body) {
        Ok((storage, loss)) => {
            let mut file_manager = file_manager.write().await;
            match file_manager.add(&storage, loss) {
                Ok(_) => Ok(StatusCode::OK),
                Err(e) => {
                    eprintln!("{}", e);
                    Ok(StatusCode::INTERNAL_SERVER_ERROR)
                }
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            Ok(StatusCode::BAD_REQUEST)
        }
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
    match file_manager.get_config() {
        Ok(config) => Ok(reply::with_status(config, StatusCode::OK)),
        Err(_) => Ok(reply::with_status(vec![], StatusCode::NOT_FOUND)),
    }
}

pub async fn post_config(body: warp::hyper::body::Bytes, file_manager: FileManagerDep) -> EndpointResult<impl Reply> {
    match load_model_xml(&body) { // Test if the xml is valid
        Ok(_) => {
            let file_manager = file_manager.read().await;
            match file_manager.set_config(&body) {
                Ok(_) => Ok(StatusCode::OK),
                Err(e) => {
                    eprintln!("{}", e);
                    Ok(StatusCode::INTERNAL_SERVER_ERROR)
                }
            }
        }
        Err(_) => Ok(StatusCode::BAD_REQUEST)
    }
}
