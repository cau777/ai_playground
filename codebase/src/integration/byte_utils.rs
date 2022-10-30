use std::{io::{self, Read}, fmt::Display, string::FromUtf8Error};

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

pub type DeserResult<T> = std::result::Result<T, StorageDeserError>;