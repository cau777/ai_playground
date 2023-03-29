use std::num::Wrapping;

pub enum BufferChecksumMethod {
    None,
    Single,
    Split,
}

// TODO: find best
pub const CHUNK_SIZE: usize = 128;

pub fn checksum_slice(slice: &[f32]) -> u64 {
    let mut result = 0;
    for (index, &num) in slice.iter().enumerate() {
        // Mask off the 4 least significant bits allowing for a degree of uncertainty
        let masked = num.to_bits() & 0xfffffff0;
        let masked = masked as u64;

        result ^= (Wrapping(masked) << index).0;
    }
    result
}
