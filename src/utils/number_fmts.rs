use std::{fs, mem};
use std::io::{BufWriter, Write, BufReader, Read};
use float_eq::float_eq;
use byteorder::{ByteOrder, LittleEndian};

pub fn integer_decode(val: f64) -> (u64, i16, i8) {
    let bits: u64 = unsafe { mem::transmute(val) };
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52 ) & 0x7ff ) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff ) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

/// This method will adjust any values close to zero as zeroes, correcting LP rounding errors
pub fn val_or_zero_one(val: &f64) -> f64 {
    if float_eq!(*val, 0., abs <= 0.25 * f64::EPSILON) {
        0.
    } else if float_eq!(*val, 1., abs <= 0.25 * f64::EPSILON) {
        1.
    } else {
        *val
    }
}

pub fn write_f64_to_file(path: &str, fname: &str, v: &[f64]) {
    let mut bytes = vec![0; v.len() * 8];
    LittleEndian::write_f64_into(v, &mut bytes[..]);
    let filename = format!("{}/{}", path, fname);
    let mut f = BufWriter::new(std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(filename.as_str())
        .expect("Couldn't open file"));
    f.write_all(&bytes[..]).expect("Unable to write rewards data");
}

pub fn read_f64_from_file(path: &str, fname: &str, size: usize) -> Vec<f64> {
    let fp = format!("{}/{}", path, fname);
    let f = fs::File::open(fp).unwrap();
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();
    // Read file into vector.
    reader.read_to_end(&mut buffer).unwrap();
    let mut numbers_returned: Vec<f64> = vec![0.0; size];
    LittleEndian::read_f64_into(&buffer, &mut numbers_returned);
    numbers_returned
}