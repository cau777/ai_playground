// use rocket::http::Method;
// use rocket_cors::{AllowedOrigins, Cors, CorsOptions};
//
// pub fn make_cors() -> Cors {
//     CorsOptions::default()
//         .allowed_origins(AllowedOrigins::all())
//         .allowed_methods(
//             vec![Method::Get, Method::Post]
//                 .into_iter()
//                 .map(From::from)
//                 .collect(),
//         )
//         .allow_credentials(true)
//         .to_cors()
//         .unwrap()
// }

use rocket::fairing::{Fairing, Info, Kind};
use rocket::http::Header;
use rocket::{Request, Response};

pub struct CORS;

#[rocket::async_trait]
impl Fairing for CORS {
    fn info(&self) -> Info {
        Info {
            name: "Add CORS headers to responses",
            kind: Kind::Response
        }
    }

    async fn on_response<'r>(&self, _request: &'r Request<'_>, response: &mut Response<'r>) {
        // println!("{:?}", _request.limits());
        response.set_header(Header::new("Access-Control-Allow-Origin", "*"));
        response.set_header(Header::new("Access-Control-Allow-Methods", "POST, GET, PATCH, OPTIONS"));
        response.set_header(Header::new("Access-Control-Allow-Headers", "*"));
        response.set_header(Header::new("Access-Control-Allow-Credentials", "true"));
    }
}