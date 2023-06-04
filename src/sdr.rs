use bitvec::prelude::*;
use rand::prelude::SliceRandom;

// NOTES: I'd like to make SDR's immutable so that I don't need to keep writing
// out "mut" everywhere. This should be doable with a RefCell lock on the whole
// structure? Or I could premptively calc both representations at construction.

// TODO: Make SDR::from_sparse() and SDR::from_dense() accept anything that can
// be coerced into an iterable.

pub type Idx = u32;

/// Sparse Distributed Representation
pub struct SDR {
    num_cells_: Idx,
    sparse_: Option<Vec<Idx>>,
    dense_: Option<BitBox>,
}

impl SDR {
    pub fn zeros(num_cells: usize) -> Self {
        return Self {
            num_cells_: num_cells.try_into().unwrap(),
            sparse_: None,
            dense_: None,
        };
    }

    pub fn ones(num_cells: usize) -> Self {
        return Self::from_dense(&vec![true; num_cells]);
    }

    pub fn random(num_cells: usize, sparsity: f32) -> Self {
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

    pub fn from_sparse(num_cells: usize, mut index: Vec<Idx>) -> Self {
        index.sort();
        index.dedup();
        if let Some(last) = index.last() {
            let last = *last as usize;
            assert!(last < num_cells);
        }
        return Self {
            num_cells_: num_cells.try_into().unwrap(),
            sparse_: Some(index),
            dense_: None,
        };
    }

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

    pub fn from_dense(dense: &[bool]) -> Self {
        let mut bits = BitVec::with_capacity(dense.len());
        for x in dense {
            bits.push(*x)
        }
        return Self {
            num_cells_: dense.len().try_into().unwrap(),
            sparse_: None,
            dense_: Some(bits.into_boxed_bitslice()),
        };
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

    pub fn overlap(&mut self, other: &mut Self) -> usize {
        assert_eq!(self.num_cells(), other.num_cells());
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

    pub fn concatenate(sdrs: &mut [SDR]) -> Self {
        let num_cells = sdrs.iter().map(|x| x.num_cells()).sum();
        let num_active = sdrs.iter_mut().map(|x| x.num_active()).sum();

        let mut sparse = Vec::with_capacity(num_active);
        let mut offset = 0;
        for x in sdrs.iter_mut() {
            for i in x.sparse() {
                sparse.push(i + offset);
            }
            offset += x.num_cells() as Idx;
        }
        return Self::from_sparse(num_cells, sparse);
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
        if let Some(sparse) = &self.sparse_ {
            writeln!(f, "SDR({}, nact={})", self.num_cells(), sparse.len())
        } else {
            writeln!(f, "SDR({})", self.num_cells(),)
        }
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

        let z = SDR::zeros(0);
        let z = SDR::ones(0);
    }

    #[test]
    fn convert() {
        let mut a = SDR::from_dense(&vec![false, true, false, true, false, true]);
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
        let mut d = SDR::concatenate(&mut [a, b, c]);
        assert_eq!(d.num_cells(), 30);
        assert_eq!(d.sparse(), &[3, 4, 5, 13, 14, 15, 23, 24, 25]);
    }

    #[test]
    fn test_union() {
        todo!();
    }

    #[test]
    fn test_intersection() {
        todo!();
    }
}
