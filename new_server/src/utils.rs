use warp::reject::Rejection;

pub type EndpointResult<T> = Result<T, Rejection>;