// TODO: Change Idx to u32.
pub type Idx = usize;

/// Sparse Distributed Representation
pub struct SDR {
    num_cells_: usize,
    sparse_: Option<Vec<Idx>>,
    dense_: Option<ndarray::Array1<bool>>,
}

impl SDR {
    pub fn zeros(num_cells: usize) -> Self {
        Self {
            num_cells_: num_cells,
            sparse_: None,
            dense_: None,
        }
    }

    pub fn ones(num_cells: usize) -> Self {
        Self::from_dense(vec![true; num_cells].into())
    }

    pub fn random(num_cells: usize, sparsity: f32) -> Self {
        let mut rng = rand::thread_rng();
        let num_active = (num_cells as f32 * sparsity).round() as usize;
        let index = rand::seq::index::sample(&mut rng, num_cells, num_active)
            .iter()
            .collect();
        Self::from_sparse(num_cells, index)
    }

    pub fn num_cells(&self) -> usize {
        self.num_cells_
    }

    pub fn num_active(&mut self) -> usize {
        self.sparse().len()
    }

    pub fn sparsity(&mut self) -> f32 {
        self.sparse().len() as f32 / self.num_cells() as f32
    }

    pub fn from_sparse(num_cells: usize, index: Vec<Idx>) -> Self {
        Self {
            num_cells_: num_cells,
            sparse_: Some(index),
            dense_: None,
        }
    }

    /// Get a read-only view of this SDR's data.
    pub fn sparse(&mut self) -> &Vec<Idx> {
        if self.sparse_.is_some() {
            // Do nothing, the data is already computed.
        } else {
            let mut index = vec![];
            if let Some(d) = &self.dense_ {
                for (i, &x) in d.iter().enumerate() {
                    if x {
                        index.push(i);
                    }
                }
            }
            self.sparse_ = Some(index);
        }
        self.sparse_.as_ref().unwrap()
    }

    /// Consume this SDR and return its sparse formatted data.
    pub fn sparse_mut(mut self) -> Vec<Idx> {
        self.sparse();
        self.sparse_.unwrap()
    }

    pub fn from_dense(dense: ndarray::Array1<bool>) -> Self {
        Self {
            num_cells_: dense.len(),
            sparse_: None,
            dense_: Some(dense),
        }
    }

    /// Get a read-only view of this SDR's data.
    pub fn dense(&mut self) -> &ndarray::Array1<bool> {
        if self.dense_.is_some() {
            // Do nothing, the data is already computed.
        } else {
            let mut d = vec![false; self.num_cells()];
            if let Some(index) = &self.sparse_ {
                for &i in index.iter() {
                    d[i] = true;
                }
            }
            self.dense_ = Some(d.into());
        }
        self.dense_.as_ref().unwrap()
    }

    /// Consume this SDR and return its dense formatted data.
    pub fn dense_mut(mut self) -> ndarray::Array1<bool> {
        self.dense();
        self.dense_.unwrap()
    }

    pub fn overlap(&mut self, other: &mut Self) -> usize {
        assert_eq!(self.num_cells(), other.num_cells());
        let mut ovlp = 0;
        for (&a, &b) in self.dense().iter().zip(other.dense()) {
            if a && b {
                ovlp += 1;
            }
        }
        ovlp
    }

    pub fn percent_overlap(&mut self, other: &mut Self) -> f32 {
        self.overlap(other) as f32 / self.num_active().max(other.num_active()) as f32
    }

    pub fn corrupt(&mut self, percent_noise: f32) -> Self {
        let index = self.sparse().clone();
        todo!();
        Self::from_sparse(self.num_cells(), index)
    }

    pub fn concatenate(sdrs: &mut [SDR]) -> Self {
        todo!()
    }

    pub fn union(sdrs: &mut [SDR]) -> Self {
        todo!()
    }

    pub fn intersection(sdrs: &mut [SDR]) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::SDR;

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
        let mut a = SDR::from_dense(vec![false, true, false, true, false, true].into());
        assert_eq!(a.sparse(), &[1, 3, 5]);

        let mut b = SDR::zeros(3);
        assert_eq!(b.sparse(), &[]);
        assert_eq!(b.dense().to_vec(), [false, false, false]);

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
    fn join() {
        // Test concatenate
        todo!();

        // Test union
        todo!();

        // Test intersection
        todo!();
    }
}
