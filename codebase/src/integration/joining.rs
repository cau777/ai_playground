use std::io::Read;

use crate::integration::byte_utils::*;
pub fn join_as_bytes(objects: &[&[u8]]) -> Vec<u8> {
    let combined_len: usize = objects.iter().map(|o| o.len()).sum();
    let mut result = Vec::with_capacity(4 + objects.len() * 4 + combined_len);
    write_u32(&mut result, objects.len() as u32);

    for obj in objects.iter() {
        write_u32(&mut result, obj.len() as u32);
        result.extend_from_slice(obj);
    }

    result
}

pub fn split_bytes<'a>(mut bytes: &'a [u8]) -> DeserResult<Vec<&'a [u8]>> {
    let count = read_u32(&mut bytes)?;
    let mut result = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let length = read_u32(&mut bytes)? as usize;
        result.push(&bytes[..length]);
        bytes = &bytes[length..];
    }

    Ok(result)
}

pub fn split_bytes_3<'a>(bytes: &'a [u8])-> DeserResult<[&'a [u8]; 3]> {
    let mut result = split_bytes(bytes)?;
    if result.len() != 3 {
        panic!("Incorrect number of arguments. Expected 3, found {}", result.len());
    }
    
    Ok([result.remove(0), result.remove(0), result.remove(0)])
}
