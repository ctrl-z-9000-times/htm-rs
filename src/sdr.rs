use bitvec::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::prelude::SliceRandom;
use std::ops::DerefMut;

pub type Idx = u32;

/// Sparse Distributed Representation
#[pyclass]
#[derive(Clone)]
pub struct SDR {
    num_cells_: Idx,
    sparse_: Option<Vec<Idx>>,
    dense_: Option<BitBox>,
}

#[pymethods]
impl SDR {
    #[staticmethod]
    pub fn zeros(num_cells: usize) -> Self {
        return Self {
            num_cells_: num_cells.try_into().unwrap(),
            sparse_: None,
            dense_: None,
        };
    }

    #[staticmethod]
    pub fn ones(num_cells: usize) -> Self {
        return Self::from_dense(vec![true; num_cells]);
    }

    #[staticmethod]
    pub fn random(num_cells: usize, sparsity: f32) -> Self {
        // TODO: Consider changing sparsity into num_active, since I think
        // that's more intuitive for the users and it more directly translates
        // into the source code.
        let mut rng = rand::thread_rng();
        let num_active = (num_cells as f32 * sparsity).round() as usize;
        let index = rand::seq::index::sample(&mut rng, num_cells, num_active)
            .iter()
            .map(|x| x as Idx)
            .collect();
        return Self::from_sparse(num_cells, index);
    }

    pub fn num_cells(&self) -> usize {
        return self.num_cells_ as usize;
    }

    pub fn num_active(&mut self) -> usize {
        return self.sparse().len();
    }

    pub fn sparsity(&mut self) -> f32 {
        if self.num_cells() == 0 {
            return 0.0;
        } else {
            return self.sparse().len() as f32 / self.num_cells() as f32;
        }
    }

    fn __str__(&self) -> String {
        return format!("{:?}", self);
    }

    #[staticmethod]
    pub fn from_sparse(num_cells: usize, mut index: Vec<Idx>) -> Self {
        index.sort();
        index.dedup();
        if let Some(last) = index.last() {
            assert!((*last as usize) < num_cells);
        }
        return Self {
            num_cells_: num_cells.try_into().unwrap(),
            sparse_: Some(index),
            dense_: None,
        };
    }

    #[staticmethod]
    pub fn from_dense(dense: Vec<bool>) -> Self {
        let num_cells = dense.len();
        let mut bits = BitVec::with_capacity(num_cells);
        for x in &dense {
            bits.push(*x)
        }
        return Self {
            num_cells_: num_cells.try_into().unwrap(),
            sparse_: None,
            dense_: Some(bits.into_boxed_bitslice()),
        };
    }

    #[pyo3(name = "sparse")]
    fn py_sparse(&mut self) -> Vec<Idx> {
        return self.sparse().clone();
    }

    #[pyo3(name = "dense")]
    fn py_dense(&mut self) -> Vec<bool> {
        return self.dense().iter().map(|x| *x).collect();
    }

    pub fn overlap(&mut self, other: &mut Self) -> usize {
        assert!(
            self.num_cells() == other.num_cells(),
            "sdr.overlap(): both SDRs must have the same num_cells"
        );
        let mut ovlp = 0;
        for (a, b) in self.dense().iter().zip(other.dense().iter()) {
            if *a && *b {
                ovlp += 1;
            }
        }
        return ovlp;
    }

    pub fn percent_overlap(&mut self, other: &mut Self) -> f32 {
        return self.overlap(other) as f32 / self.num_active().max(other.num_active()) as f32;
    }

    pub fn corrupt(&mut self, percent_noise: f32) -> Self {
        assert!(0.0 <= percent_noise && percent_noise <= 1.0);
        let num_cells = self.num_cells();
        let num_active = self.num_active();
        let active = self.sparse();
        // Make a list of all cells that are not active.
        let mut silent = Vec::with_capacity(num_cells - num_active);
        let mut active_iter = active.iter().peekable();
        for cell in 0..num_cells as Idx {
            if active_iter.peek() == Some(&&cell) {
                active_iter.next();
            } else {
                silent.push(cell);
            }
        }
        // Choose the cells to move and where to move them.
        let mut rng = rand::thread_rng();
        let num_move = (percent_noise * active.len() as f32).round() as usize;
        assert!(num_move <= active.len());
        assert!(num_move <= silent.len());
        let turn_off = active.choose_multiple(&mut rng, num_move);
        let turn_on = silent.choose_multiple(&mut rng, num_move);
        // Build the new SDR's sparse data list.
        let mut corrupted = Vec::with_capacity(active.len());
        let mut turn_off: Vec<_> = turn_off.collect();
        turn_off.sort();
        let mut turn_off_iter = turn_off.iter().peekable();
        for cell in active {
            if turn_off_iter.peek() == Some(&&cell) {
                turn_off_iter.next();
            } else {
                corrupted.push(*cell);
            }
        }
        for cell in turn_on {
            corrupted.push(*cell);
        }
        return Self::from_sparse(self.num_cells(), corrupted);
    }

    #[staticmethod]
    #[pyo3(name = "concatenate")]
    #[args(sdrs = "*")]
    fn py_concatenate(sdrs: &PyTuple) -> Self {
        let mut sdrs: Vec<PyRefMut<SDR>> = sdrs.iter().map(|arg: &PyAny| arg.extract().unwrap()).collect();
        let mut sdrs: Vec<&mut SDR> = sdrs.iter_mut().map(|arg| arg.deref_mut()).collect();
        return SDR::concatenate(sdrs.as_mut_slice());
    }
}

impl SDR {
    /// Get a read-only view of this SDR's data.
    pub fn sparse(&mut self) -> &Vec<Idx> {
        if self.sparse_.is_none() {
            let mut index = vec![];
            if let Some(d) = &self.dense_ {
                for (i, x) in d.iter().enumerate() {
                    if *x {
                        index.push(i as Idx);
                    }
                }
            }
            self.sparse_ = Some(index);
        }
        return self.sparse_.as_ref().unwrap();
    }

    /// Consume this SDR and return its sparse formatted data.
    pub fn sparse_mut(mut self) -> Vec<Idx> {
        self.sparse();
        return self.sparse_.unwrap();
    }

    /// Get a read-only view of this SDR's data.
    pub fn dense(&mut self) -> &BitBox {
        if self.dense_.is_none() {
            let mut bits = bitvec![0; self.num_cells()];
            if let Some(index) = &self.sparse_ {
                for i in index.iter() {
                    bits.set(*i as usize, true);
                }
            }
            self.dense_ = Some(bits.into_boxed_bitslice());
        }
        return self.dense_.as_ref().unwrap();
    }

    /// Consume this SDR and return its dense formatted data.
    pub fn dense_mut(mut self) -> BitBox {
        self.dense();
        return self.dense_.unwrap();
    }

    pub fn concatenate(sdrs: &mut [&mut SDR]) -> Self {
        let num_cells = sdrs.iter().map(|x| x.num_cells()).sum::<usize>();
        let num_active = sdrs.iter_mut().map(|x| x.num_active()).sum();

        let mut sparse = Vec::with_capacity(num_active);
        let mut offset = 0;
        for x in sdrs.iter_mut() {
            for i in x.sparse() {
                sparse.push(i + offset);
            }
            offset += x.num_cells() as Idx;
        }
        return Self {
            num_cells_: num_cells.try_into().unwrap(),
            sparse_: Some(sparse),
            dense_: None,
        };
    }

    pub fn union(sdrs: &mut [SDR]) -> Self {
        return todo!();
    }

    pub fn intersection(sdrs: &mut [SDR]) -> Self {
        return todo!();
    }
}

impl std::fmt::Debug for SDR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SDR({} cells, ", self.num_cells(),)?;
        if let Some(sparse) = &self.sparse_ {
            write!(f, "{} active, ", sparse.len(),)?;
        } else {
            write!(f, "? active, ",)?;
        }
        match (self.sparse_.is_some(), self.dense_.is_some()) {
            (false, false) => write!(f, "--)")?,
            (false, true) => write!(f, "-d)")?,
            (true, false) => write!(f, "s-)")?,
            (true, true) => write!(f, "sd)")?,
        }
        Ok(())
    }
}
// Display message is the same as Debug message.
impl std::fmt::Display for SDR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

///
#[pyclass]
pub struct Stats {
    num_cells_: Idx,
    num_samples_: usize,
    period_: f32,

    frequencies_: Vec<f32>,

    min_sparsity_: f32,
    max_sparsity_: f32,
    mean_sparsity_: f32,
    var_sparsity_: f32,

    previous_sdr_: SDR,

    min_overlap_: f32,
    max_overlap_: f32,
    mean_overlap_: f32,
    var_overlap_: f32,
}

#[pymethods]
impl Stats {
    #[new]
    pub fn new(num_cells: usize, period: f32) -> Self {
        return Stats {
            num_cells_: num_cells as Idx,
            num_samples_: 0,
            period_: period,
            frequencies_: vec![0.0; num_cells],
            min_sparsity_: f32::NAN,
            max_sparsity_: f32::NAN,
            mean_sparsity_: f32::NAN,
            var_sparsity_: f32::NAN,
            previous_sdr_: SDR::zeros(num_cells),
            min_overlap_: f32::NAN,
            max_overlap_: f32::NAN,
            mean_overlap_: f32::NAN,
            var_overlap_: f32::NAN,
        };
    }

    pub fn update(&mut self, sdr: &mut SDR) {
        if self.num_samples_ == 0 {
            self.mean_sparsity_ = 0.0;
            self.var_sparsity_ = 0.0;
            self.mean_overlap_ = 0.0;
            self.var_overlap_ = 0.0;
        }
        let decay = (-1.0 / self.period_.min(self.num_samples_ as f32)).exp();
        let alpha = 1.0 - decay;
        self.num_samples_ += 1;

        // Update the activation frequency data.
        for (frq, state) in self.frequencies_.iter_mut().zip(sdr.dense().iter()) {
            *frq += alpha * (*state as usize as f32 - *frq);
        }

        // Update the sparsity statistics.
        let sparsity = sdr.sparsity();
        self.min_sparsity_ = self.min_sparsity_.min(sparsity);
        self.max_sparsity_ = self.max_sparsity_.max(sparsity);
        // http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
        // See section 9.
        let diff = sparsity - self.mean_sparsity_;
        let incr = alpha * diff;
        self.mean_sparsity_ += incr;
        self.var_sparsity_ = decay * (self.var_sparsity_ + diff * incr);

        // Update the sequential overlap statistics.
        let overlap = sdr.percent_overlap(&mut self.previous_sdr_);
        self.previous_sdr_ = sdr.clone();
        self.min_overlap_ = self.min_overlap_.min(overlap);
        self.max_overlap_ = self.max_overlap_.max(overlap);
        let diff = overlap - self.mean_overlap_;
        let incr = alpha * diff;
        self.mean_overlap_ += incr;
        self.var_overlap_ = decay * (self.var_overlap_ + diff * incr);
    }

    pub fn reset(&mut self) {
        self.previous_sdr_ = SDR::zeros(self.num_cells());
    }

    pub fn num_cells(&self) -> usize {
        return self.num_cells_ as usize;
    }
    pub fn num_samples(&self) -> usize {
        return self.num_samples_ as usize;
    }
    pub fn min_sparsity(&self) -> f32 {
        return self.min_sparsity_;
    }
    pub fn max_sparsity(&self) -> f32 {
        return self.max_sparsity_;
    }
    pub fn mean_sparsity(&self) -> f32 {
        return self.mean_sparsity_;
    }
    pub fn std_sparsity(&self) -> f32 {
        return self.var_sparsity_.sqrt();
    }
    pub fn min_frequency(&self) -> f32 {
        if self.num_samples_ == 0 {
            return f32::NAN;
        } else {
            return self.frequencies_.iter().fold(f32::NAN, |a, &b| a.min(b));
        }
    }
    pub fn max_frequency(&self) -> f32 {
        if self.num_samples_ == 0 {
            return f32::NAN;
        } else {
            return self.frequencies_.iter().fold(f32::NAN, |a, &b| a.max(b));
        }
    }
    pub fn mean_frequency(&self) -> f32 {
        if self.num_samples_ == 0 {
            return f32::NAN;
        } else {
            return self.frequencies_.iter().sum::<f32>() / self.num_cells_ as f32;
        }
    }
    pub fn std_frequency(&self) -> f32 {
        if self.num_samples_ == 0 {
            return f32::NAN;
        } else {
            let mean = self.mean_frequency();
            return self.frequencies_.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.num_cells_ as f32;
        }
    }
    pub fn entropy(&self) -> f32 {
        let max_extropy = Stats::binary_entropy(&[self.mean_frequency()]);
        if max_extropy == 0.0 {
            return 0.0;
        } else {
            return Stats::binary_entropy(&self.frequencies_) / max_extropy;
        }
    }
    pub fn min_overlap(&self) -> f32 {
        return self.min_overlap_;
    }
    pub fn max_overlap(&self) -> f32 {
        return self.max_overlap_;
    }
    pub fn mean_overlap(&self) -> f32 {
        return self.mean_overlap_;
    }
    pub fn std_overlap(&self) -> f32 {
        return self.var_overlap_.sqrt();
    }
    fn __str__(&self) -> String {
        return format!("{}", self);
    }
}

impl Stats {
    fn binary_entropy(data: &[f32]) -> f32 {
        return data
            .iter()
            .map(|p| {
                let p_ = 1.0 - p;
                let e = -p * p.log2() - p_ * p_.log2();
                if e.is_nan() {
                    0.0
                } else {
                    e
                }
            })
            .sum::<f32>()
            / data.len() as f32;
    }
}

impl std::fmt::Debug for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "SDR Statistics({} cells, {} samples)",
            self.num_cells(),
            self.num_samples()
        )?;
        writeln!(f, "           |  min  |  max  |  mean |  std  |",)?;
        writeln!(
            f,
            "Sparsity   | {:.3} | {:.3} | {:.3} | {:.3} |",
            self.min_sparsity(),
            self.max_sparsity(),
            self.mean_sparsity(),
            self.std_sparsity()
        )?;
        writeln!(
            f,
            "Frequency  | {:.3} | {:.3} | {:.3} | {:.3} |",
            self.min_frequency(),
            self.max_frequency(),
            self.mean_frequency(),
            self.std_frequency()
        )?;
        writeln!(
            f,
            "Overlap    | {:.3} | {:.3} | {:.3} | {:.3} |",
            self.min_overlap(),
            self.max_overlap(),
            self.mean_overlap(),
            self.std_overlap()
        )?;
        writeln!(f, "Entropy: {:.1}%", (self.entropy() * 100.0))
    }
}
// Display message is the same as Debug message.
impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut a = SDR::zeros(111);
        assert_eq!(a.num_cells(), 111);
        assert_eq!(a.sparsity(), 0.0);

        let mut b = SDR::ones(33);
        assert_eq!(b.num_cells(), 33);
        assert_eq!(b.sparsity(), 1.0);
        assert_eq!(b.clone().sparse(), b.sparse());

        let z = SDR::zeros(0);
        let z = SDR::ones(0);
    }

    #[test]
    fn convert() {
        let mut a = SDR::from_dense(vec![false, true, false, true, false, true]);
        assert_eq!(a.sparse(), &[1, 3, 5]);

        let mut b = SDR::zeros(3);
        assert_eq!(b.sparse(), &[]);
        assert_eq!(b.dense(), bits![0, 0, 0]);

        let mut c = SDR::from_sparse(6, vec![1, 3, 5]);
        assert_eq!(c.dense(), a.dense());
    }

    #[test]
    fn overlap() {
        let mut a = SDR::from_sparse(100, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut b = SDR::from_sparse(100, vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
        assert_eq!(a.overlap(&mut b), 5);
        assert_eq!(a.percent_overlap(&mut b), 0.5);
    }

    #[test]
    fn random() {
        let mut a = SDR::random(100, 0.1);
        assert_eq!(a.num_cells(), 100);
        assert_eq!(a.sparsity(), 0.1);

        let mut b = SDR::random(100, 0.1);
        assert!(a.dense() != b.dense());
    }

    #[test]
    fn corrupt() {
        let mut a = SDR::random(100, 0.1);
        let mut b = a.corrupt(0.5);
        assert_eq!(a.overlap(&mut b), 5);
    }

    #[test]
    fn concatenate() {
        let mut a = SDR::from_sparse(10, vec![3, 4, 5]);
        let mut b = SDR::from_sparse(10, vec![3, 4, 5]);
        let mut c = SDR::from_sparse(10, vec![3, 4, 5]);
        let mut d = SDR::concatenate(&mut [&mut a, &mut b, &mut c]);
        assert_eq!(d.num_cells(), 30);
        assert_eq!(d.sparse(), &[3, 4, 5, 13, 14, 15, 23, 24, 25]);
    }

    // #[test]
    fn test_union() {
        todo!();
    }

    // #[test]
    fn test_intersection() {
        todo!();
    }
}
