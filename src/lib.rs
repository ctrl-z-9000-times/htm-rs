#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(non_camel_case_types)]
#![allow(clippy::needless_return)]

mod sdr;
pub use sdr::{Idx, Stats, SDR};

mod syn;
pub use syn::Synapses;

mod sp;
pub use sp::{SpatialPooler, SpatialPoolerParameters};

// mod tm;
// pub use tm::TemporalMemory;

use pyo3::prelude::*;

///
#[pymodule]
fn htm_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SDR>()?;
    m.add_class::<Stats>()?;
    m.add_class::<SpatialPoolerParameters>()?;
    m.add_class::<SpatialPooler>()?;

    Ok(())
}
