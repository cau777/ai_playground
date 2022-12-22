use std::{io::{self, Read}, fmt::Display, string::FromUtf8Error, iter};
use ndarray::{ArrayViewD, Axis, concatenate, Slice};
use ndarray_rand::rand;
use ndarray_rand::rand::Rng;
use crate::ArrayDynF;
use crate::integration::random_picker::RandomPicker;

pub fn read_u8(source: &mut &[u8]) -> io::Result<u8> {
    let mut buffer = [0];
    source.read_exact(&mut buffer)?;
    Ok(buffer[0])
}

pub fn read_u32(source: &mut &[u8]) -> io::Result<u32> {
    let mut buffer = [0; 4];
    source.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

pub fn write_u32(result: &mut Vec<u8>, num: u32) {
    result.extend(num.to_be_bytes())
}

pub fn read_f64(source: &mut &[u8]) -> io::Result<f64> {
    let mut buffer = [0; 8];
    source.read_exact(&mut buffer)?;
    Ok(f64::from_be_bytes(buffer))
}

pub fn write_f64(result: &mut Vec<u8>, num: f64) {
    result.extend(num.to_be_bytes())
}

#[derive(Debug)]
enum ErrorKind {
    NotEnoughBytes,
    WrongStringEncoding(FromUtf8Error),
}

#[derive(Debug)]
pub struct StorageDeserError {
    kind: ErrorKind,
}

impl StorageDeserError {
    fn new(kind: ErrorKind) -> Self {
        Self { kind }
    }
}

impl Display for StorageDeserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<io::Error> for StorageDeserError {
    fn from(_: io::Error) -> Self {
        Self::new(ErrorKind::NotEnoughBytes)
    }
}

impl From<FromUtf8Error> for StorageDeserError {
    fn from(err: FromUtf8Error) -> Self {
        Self::new(ErrorKind::WrongStringEncoding(err))
    }
}

impl std::error::Error for StorageDeserError {}

pub type DeserResult<T> = Result<T, StorageDeserError>;

#[derive(Debug)]
pub struct Pairs {
    pub inputs: ArrayDynF,
    pub expected: ArrayDynF,
}

impl Pairs {
    pub fn pick_rand(&self, count: usize, rng: &mut impl rand::RngCore) -> Pairs {
        let total = self.inputs.shape()[0];
        let mut picker = RandomPicker::new(total);
        let mut new_inputs = Vec::with_capacity(count);
        let mut new_expected = Vec::with_capacity(count);

        for _ in 0..count {
            let chosen = picker.pick(rng);
            new_inputs.push(self.inputs.slice_axis(Axis(0), Slice::from(chosen..chosen + 1)));
            new_expected.push(self.expected.slice_axis(Axis(0), Slice::from(chosen..chosen + 1)));
        }

        Pairs {
            inputs: concatenate(Axis(0), &new_inputs).unwrap(),
            expected: concatenate(Axis(0), &new_expected).unwrap(),
        }
    }

    pub fn chunks_iter(&self, size: usize) -> impl Iterator<Item = (ArrayViewD<f32>, ArrayViewD<f32>)> {
        iter::zip(self.inputs.axis_chunks_iter(Axis(0), size),
        self.expected.axis_chunks_iter(Axis(0), size))
    }
}
