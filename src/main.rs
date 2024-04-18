#![allow(non_snake_case)]

mod sparse;

use std::{
    fs::{self, File},
    io::BufReader,
    sync::Mutex,
    time::Instant,
};

use camino::{Utf8Path, Utf8PathBuf};
use halo2curves::bn256;
use once_cell::sync::OnceCell;
use serde::de::DeserializeOwned;
use sparse::SparseMatrix;

/// Path to the directory where Arecibo data will be stored.
pub static ARECIBO_DATA: &str = ".arecibo_data";

/// Global configuration for Arecibo data storage, including root directory and counters.
/// This configuration is initialized on first use.
pub static ARECIBO_CONFIG: OnceCell<Mutex<DataConfig>> = OnceCell::new();

/// Configuration for managing Arecibo data files, including the root directory,
/// witness counter, and cross-term counter for organizing files.
#[derive(Debug, Clone, Default)]
pub struct DataConfig {
    root_dir: Utf8PathBuf,
}

pub fn init_config() -> Mutex<DataConfig> {
    let root_dir = home::home_dir().unwrap().join(ARECIBO_DATA);
    let root_dir = Utf8PathBuf::from_path_buf(root_dir).unwrap();
    if !root_dir.exists() {
        fs::create_dir_all(&root_dir).expect("Failed to create arecibo data directory");
    }

    let config = DataConfig { root_dir };

    Mutex::new(config)
}

pub fn read_arecibo_data<T: DeserializeOwned>(
    section: impl AsRef<Utf8Path>,
    label: impl AsRef<Utf8Path>,
) -> T {
    let mutex = ARECIBO_CONFIG.get_or_init(init_config);
    let config = mutex.lock().unwrap();

    let section_path = config.root_dir.join(section.as_ref());
    assert!(
        section_path.exists(),
        "Section directory does not exist: {}",
        section_path
    );

    // Assuming the label uniquely identifies the file, and ignoring the counter for simplicity
    let file_path = section_path.join(label.as_ref());
    assert!(
        file_path.exists(),
        "Data file does not exist: {}",
        file_path
    );

    let file = File::open(file_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).expect("Failed to read data")
}

/// cargo run --release -- <HASH>
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        eprintln!("usage: {} <HASH>", args[0]);
        return;
    }

    let hash = args[1].to_string();

    let A: SparseMatrix<bn256::Fr> = read_arecibo_data(format!("sparse_matrices_{}", hash), "A_0");
    let B: SparseMatrix<bn256::Fr> = read_arecibo_data(format!("sparse_matrices_{}", hash), "B_0");
    let C: SparseMatrix<bn256::Fr> = read_arecibo_data(format!("sparse_matrices_{}", hash), "C_0");

    for i in 0..16 {
        println!("timing: {i}");
        let witness: Vec<bn256::Fr> =
            read_arecibo_data(format!("witness_{}", hash), format!("_{}", i));

        let start = Instant::now();
        let AZ = A.multiply_vec(&witness);
        let AZ_time = start.elapsed();
        println!("AZ took: {:?}", AZ_time);

        let BZ = B.multiply_vec(&witness);
        let BZ_time = start.elapsed();
        println!("BZ took: {:?}", BZ_time);

        let CZ = C.multiply_vec(&witness);
        let CZ_time = start.elapsed();
        println!("CZ took: {:?}", CZ_time);
        println!();

        let AZ_expected: Vec<bn256::Fr> =
            read_arecibo_data(format!("result_{}", hash), format!("AZ_{}", i));
        let BZ_expected: Vec<bn256::Fr> =
            read_arecibo_data(format!("result_{}", hash), format!("BZ_{}", i));
        let CZ_expected: Vec<bn256::Fr> =
            read_arecibo_data(format!("result_{}", hash), format!("CZ_{}", i));

        assert_eq!(AZ, AZ_expected);
        assert_eq!(BZ, BZ_expected);
        assert_eq!(CZ, CZ_expected);
    }
}
