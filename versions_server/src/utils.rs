use std::fmt;
use warp::{Rejection, reply, Reply};
use warp::reply::WithStatus;
use crate::StatusCode;

pub type EndpointResult<T> = Result<T, Rejection>;

pub fn stderr_proc(e: impl fmt::Debug) -> EndpointResult<StatusCode> {
    eprintln!("{:?}", e);
    Ok(StatusCode::INTERNAL_SERVER_ERROR)
}

pub fn data_err_proc<T: Reply>(e: impl fmt::Debug, empty_message: T) -> EndpointResult<WithStatus<T>> {
    eprintln!("{:?}", e);
    Ok(reply::with_status(empty_message, StatusCode::INTERNAL_SERVER_ERROR))
}