use std::fs::OpenOptions;
use std::io;
use std::io::ErrorKind::InvalidData;
use std::io::Read;
use codebase::integration::deserialization::deserialize_pairs;
use codebase::integration::serde_utils::Pairs;
use crate::EnvConfig;

pub fn load_file_data(filename: &str,name: &str, config: &EnvConfig) -> io::Result<Pairs> {
    let mut file = OpenOptions::new().read(true)
        .open(format!("{}/{}/{}.dat", config.mounted_path, name, filename))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    deserialize_pairs(&buffer).map_err(|e| io::Error::new(InvalidData, e))
}

pub fn load_file_lines<'a>(filename: &str,name: &str, config: &EnvConfig, buffer: &'a mut String) -> io::Result<Vec<&'a str>> {
    let mut file = OpenOptions::new().read(true)
        .open(format!("{}/{}/{}.dat", config.mounted_path, name, filename))?;
    file.read_to_string(buffer)?;
    Ok(buffer.lines().collect())
}