use std::io;
use std::io::{Read, Write};
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

pub fn compress_default(bytes: &[u8]) -> Result<Vec<u8>, io::Error> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(bytes)?;
    encoder.finish()
}

pub fn decompress_default(bytes: &[u8]) -> Result<Vec<u8>, io::Error> {
    let mut result = Vec::new();
    let mut decoder = ZlibDecoder::new(bytes);
    decoder.read_to_end(&mut result)?;
    Ok(result)
}