#![allow(warnings, unused)] // Silence.
#![allow(clippy::needless_return)]

mod sdr;
pub use sdr::{Idx, Stats, SDR};

mod syn;
pub use syn::Synapses;

mod sp;
pub use sp::SpatialPooler;

mod enc;
pub use enc::Encoder;

mod cerebellum;
pub use cerebellum::Cerebellum;

// mod tm;
// pub use tm::TemporalMemory;

use pyo3::prelude::*;

///
#[pymodule]
fn htm_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SDR>()?;
    m.add_class::<Stats>()?;
    m.add_class::<SpatialPooler>()?;
    m.add_class::<Encoder>()?;
    m.add_class::<Cerebellum>()?;

    Ok(())
}
