#![allow(non_snake_case)]

extern crate cblas_sys;

pub mod scpm;
pub mod utils;
pub mod algorithm;
pub mod c_binding;
pub mod definition;
pub mod solver;
pub mod agent;
pub mod dfa;

use std::hash::Hash;
use std::collections::HashMap;
use std::fs;
use std::io::BufWriter;
use cblas_sys::{cblas_dcopy, cblas_dscal, cblas_ddot, cblas_dgemv};
use c_binding::suite_sparse::*;
use serde::{Serialize, Deserialize};
use hashbrown::HashMap as FastHM;
use crate::utils::number_fmts::read_f64_from_file;

pub fn create_sparse_matrix(m: i32, n: i32, rows: &[i32], cols: &[i32], x: &[f64])
                            -> *mut cs_di {
    unsafe {
        let T: *mut cs_di = cs_di_spalloc(m, n, x.len() as i32, 1, 1);
        for (k, elem) in x.iter().enumerate() {
            cs_di_entry(T, rows[k], cols[k], *elem);
        }
        return T
    }
}

pub fn convert_to_compressed(T: *mut cs_di) -> *mut cs_di {
    unsafe {
        cs_di_compress(T)
    }
}

pub fn print_matrix(A: *mut cs_di) {
    unsafe {
        cs_di_print(A, 0);
    }
}

pub fn transpose(A: *mut cs_di, nnz: i32) -> *mut cs_di {
    unsafe {
        cs_di_transpose(A, nnz)
    }
}

pub fn sp_mm_multiply_f64(A: *mut cs_di, B: *mut cs_di) -> *mut cs_di {
    unsafe {
        cs_di_multiply(A, B)
    }
}

pub fn sp_mv_multiply_f64(A: *mut cs_di, x: &[f64], y: &mut [f64]) -> i32 {
    unsafe {
        cs_di_gaxpy(A, x.as_ptr(), y.as_mut_ptr())
    }
}

pub fn spfree(A: *mut cs_di) {
    unsafe {
        cs_di_spfree(A);
    }
}

pub fn spalloc(m: i32, n: i32, nzmax: i32, values: i32, t: i32) -> *mut cs_di {
    unsafe {
        cs_di_spalloc(m, n, nzmax, values, t)
    }
}

#[allow(non_snake_case)]
pub fn add_vecs(x: &[f64], y: &mut [f64], ns: i32, alpha: f64) {
    unsafe {
        cblas_sys::cblas_daxpy(ns, alpha, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

pub fn copy(x: &[f64], y: &mut [f64], ns: i32) {
    unsafe {
        cblas_dcopy(ns, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

pub fn dscal(x: &mut [f64], ns: i32, alpha: f64) {
    unsafe {
        cblas_dscal(ns, alpha, x.as_mut_ptr(), 1);
    }
}

fn blas_matrix_vector_mulf64(matrix: &[f64], v: &[f64], m: i32, n: i32, result: &mut [f64]) {
    unsafe {
        cblas_dgemv(
            cblas_sys::CBLAS_LAYOUT::CblasColMajor,
            cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            1.0,
            matrix.as_ptr(),
            m,
            v.as_ptr(),
            1,
            1.0,
            result.as_mut_ptr(),
            1
        )
    }
}

fn fast_reverse_key_value_pairs<S>(m: &FastHM<S, usize>) -> FastHM<usize, S>
    where S: Clone {
    m.into_iter()
        .fold(
            FastHM::new(),
            |mut acc, (a, b)|
                {
                    acc.insert(*b, a.clone() );
                    acc
                }
        )
}

fn blas_dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    unsafe {
        cblas_ddot(v1.len() as i32, v1.as_ptr(), 1, v2.as_ptr(), 1)
    }
}

fn reverse_key_value_pairs<S, T>(m: &FastHM<S, T>) -> FastHM<T, S>
    where S: Clone, T: Copy + Hash + Eq{
    m.into_iter()
        .fold(
            FastHM::new(),
            |mut acc, (a, b)|
                {
                    acc.insert(*b, a.clone() );
                    acc
                }
        )
}

pub struct SparseMatrixComponents {
    pub i: Vec<i32>, // row indices per column
    pub p: Vec<i32>, // column ranges
    pub x: Vec<f64>  // values per column row indices
}

pub fn deconstruct(A: *mut cs_di, nnz: usize, cols: usize) -> SparseMatrixComponents {
    let x: Vec<f64>;
    let p: Vec<i32>;
    let i: Vec<i32>;
    unsafe {
        x = Vec::from_raw_parts((*A).x as *mut f64, nnz, nnz);
        i = Vec::from_raw_parts((*A).i as *mut i32, nnz, nnz);
        p = Vec::from_raw_parts((*A).p as *mut i32, cols + 1, cols + 1);
    }
    SparseMatrixComponents {i, p, x}
}

#[derive(Debug, Serialize, Deserialize)]
// We should be able to run Rayon with this structure
pub struct Sparse {
    pub nzmax: i32,
    pub nr: i32,
    pub nc: i32,
    pub p: Vec<i32>,
    pub i: Vec<i32>,
    pub x: Vec<f64>,
    pub nz: i32,
}

pub fn cs_to_rust_and_destroy(A: *mut cs_di, nnz: i32, m: i32, n: i32) -> Sparse {
    let x: Vec<f64>;
    let p: Vec<i32>;
    let i: Vec<i32>;
    unsafe {
        x = Vec::from_raw_parts((*A).x as *mut f64, nnz as usize, nnz as usize);
        i = Vec::from_raw_parts((*A).i as *mut i32, nnz as usize, nnz as usize);
        p = Vec::from_raw_parts((*A).p as *mut i32, m as usize, m as usize);
        //println!("Deconstruction:\ni: {:?}\np: {:?}\nx: {:?}", i, p, x);
        //cs_di_spfree(A);
    }
    Sparse {
        nzmax: nnz + 1,
        nr: m,
        nc: n,
        p,
        i,
        x,
        nz: nnz
    }
}

impl Sparse {
    pub fn store_matrix_as_yaml(&self, filename: &str) {
        let mut path = std::env::var("SCPM_HOME").unwrap();
        path.push_str(filename);
        //let path_str = String::from(path.to_string_lossy());
        match fs::remove_file(path.as_str()) {
            Ok(_) => {
                // the file has been removed
            }
            Err(_) => {
                // no file exists which is fine
            }
        }
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(path.as_str())
            .expect("Couldn't open file");
        serde_yaml::to_writer(f, &self).unwrap();
    }

    pub fn read_matrix_from_file(filename: &str) -> Sparse {
        //println!("opening: {:?}", filename);
        let mut path = std::env::var("SCPM_HOME").unwrap();
        path.push_str(filename);
        let f = std::fs::File::open(path.as_str()).expect("Error opening file");
        let sparse: Sparse = serde_yaml::from_reader(f).expect("Could not read yaml into sparse");
        sparse
    }
}
#[derive(Debug, Serialize, Deserialize)]
// We should be able to run Rayon with this structure
pub struct COO {
    pub nzmax: i32,
    pub nr: i32,
    pub nc: i32,
    pub i: Vec<i32>,
    pub j: Vec<i32>,
    pub x: Vec<f64>,
    pub nz: i32,
}

impl COO {
    pub fn store_matrix_as_yaml(&self, filename: &str) {
        let mut path = std::env::var("SCPM_HOME").unwrap();
        let fp = format!("transitions/{}", filename);
        path.push_str(fp.as_str());
        // recursively check that the directory exists
        match fs::remove_file(path.as_str()) {
            Ok(_) => {
                // the file has been removed
            }
            Err(_) => {
                // no file exists which is fine
            }
        }
        let f = BufWriter::new(std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(path.as_str())
            .expect("Couldn't open file"));
        serde_yaml::to_writer(f, &self).unwrap();
        //println!("finished writing: {}", filename);
    }

    pub fn read_matrix_from_file(filename: &str) -> COO {
        let mut path = std::env::var("SCPM_HOME").unwrap();
        let fp = format!("transitions/{}", filename);
        path.push_str(fp.as_str());
        //println!("opening: {:?}", filename);
        let f = std::fs::File::open(path).expect("Error opening file");
        let triple: COO = serde_yaml::from_reader(f).expect("Could not read yaml into sparse");
        triple
    }
}

pub fn read_rewards_matrix(filename: &str, nr: usize, nobjs: usize) -> Vec<f64> {
    let mut path = std::env::var("SCPM_HOME").unwrap();
    let fp = format!("rewards");
    path.push_str(fp.as_str());
    read_f64_from_file(path.as_str(), filename, nr * nobjs)
}

pub fn vec_compare<T: std::cmp::PartialEq>(v: &[T]) -> Vec<usize> {
    let vlen: usize = v.len();
    let mut equal: Vec<usize> = Vec::new();
    for (ii, x) in v.iter().enumerate() {
        for jj in 0..vlen {
            if ii != jj {
                if *x == v[jj] {
                    equal.push(ii);
                }
            }
        }
    }
    equal
}

pub fn duplicates_equal_to<T: std::cmp::PartialEq>(v: &[T], val: &T) -> Vec<usize> {
    v.iter().enumerate().filter(|(_i, x)| *x == val).map(|(i, _x)| i).collect::<Vec<usize>>()
}

pub fn sum_first_k_elems(hullset: &HashMap<usize, Vec<f64>>, k: usize) -> Vec<f64> {
    hullset.iter().map(|(_i, x)| {
        x[..k].iter().fold(0., |acc, z| acc + *z)
    }).collect::<Vec<f64>>()
}

/// Converts a Sparse struct representing a matrix into a C struct for CSS Sparse matrix
/// the C struct doesn't really exist, it is a mutable pointer reference to the Sparse struct
pub fn sparse_to_cs(sparse: &COO) -> *mut cs_di {
    let T = create_sparse_matrix(
        sparse.nr,
        sparse.nc,
        &sparse.i[..],
        &sparse.j[..],
        &sparse.x[..]
    );
    convert_to_compressed(T)
}

///
pub struct ChannelMetaData {
    pub A: COO,
    pub R: Vec<f64>,
    pub a: i32,
    pub t: i32,
    pub act: i32
}

#[derive(Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskAgent {
    pub a: i32,
    pub t: i32
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
