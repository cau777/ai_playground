use std::path::{Path, PathBuf};

fn main() {
    protobuf_codegen::Codegen::new()
        .protoc()
        .protoc_path(&protoc_bin_vendored::protoc_bin_path().unwrap_or_else(|_| PathBuf::from(Path::new("/"))))
        .includes(&["src/protos"])
        .input("src/protos/model_storage.proto")
        .out_dir("src/compiled_protos")
        .run_from_script();
}