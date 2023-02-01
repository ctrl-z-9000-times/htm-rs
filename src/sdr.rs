// TODO: Consider changing these to u32 and f32...
pub type Idx = usize;
pub type AP = u8;

///
pub struct SDR {
    len_: usize,
    sparse_: Option<(Vec<Idx>, Vec<AP>)>,
    dense_: Option<ndarray::Array1<AP>>,
}

impl SDR {
    ///
    pub fn len(&self) -> usize {
        self.len_
    }

    ///
    pub fn is_empty(&self) -> bool {
        self.len_ == 0
    }

    /// Zero-init
    pub fn new(len: usize) -> Self {
        Self {
            len_: len,
            sparse_: None,
            dense_: None,
        }
    }

    ///
    pub fn from_sparse(len: usize, index: Vec<Idx>, activity: Option<Vec<AP>>) -> Self {
        let activity = activity.unwrap_or_else(|| vec![1; len]);
        Self {
            len_: len,
            sparse_: Some((index, activity)),
            dense_: None,
        }
    }

    /// Read-only view of this SDR data.
    pub fn sparse(&mut self) -> &(Vec<Idx>, Vec<AP>) {
        if self.sparse_.is_some() {
            // Do nothing, the data is already computed.
        } else {
            let mut index = vec![];
            let mut activity = vec![];
            if let Some(d) = &self.dense_ {
                for (i, &x) in d.iter().enumerate() {
                    if x > 0 {
                        index.push(i);
                        activity.push(x);
                    }
                }
            }
            self.sparse_ = Some((index, activity));
        }
        self.sparse_.as_ref().unwrap()
    }

    /// Consumes this SDR and returns its sparse formatted data.
    pub fn sparse_mut(mut self) -> (Vec<Idx>, Vec<AP>) {
        self.sparse();
        self.sparse_.unwrap()
    }

    ///
    pub fn from_dense(dense: ndarray::Array1<AP>) -> Self {
        Self {
            len_: dense.len(),
            sparse_: None,
            dense_: Some(dense),
        }
    }

    /// Read-only view of this SDR's data.
    pub fn dense(&mut self) -> &ndarray::Array1<AP> {
        if self.dense_.is_some() {
            // Do nothing, the data is already computed.
        } else {
            let mut d = vec![0; self.len()];
            if let Some((index, activity)) = &self.sparse_ {
                for (&i, &x) in index.iter().zip(activity) {
                    d[i] = x;
                }
            }
            self.dense_ = Some(d.into());
        }
        self.dense_.as_ref().unwrap()
    }

    /// Consumes this SDR and returns its data.
    pub fn dense_mut(mut self) -> ndarray::Array1<AP> {
        self.dense();
        self.dense_.unwrap()
    }

    ///
    pub fn sparsity(&mut self) -> f32 {
        self.sparse().0.len() as f32 / self.len() as f32
    }

    ///
    pub fn ones(len: usize) -> Self {
        Self::from_dense(vec![1; len].into())
    }

    ///
    pub fn random(len: usize, sparsity: f32) -> Self {
        let mut rng = rand::thread_rng();
        let active = (len as f32 * sparsity + 0.5) as usize;
        let index = rand::seq::index::sample(&mut rng, len, active);
        let index = index.iter().collect();
        Self::from_sparse(len, index, None)
    }

    ///
    pub fn corrupt() {
        todo!()
    }

    ///
    pub fn overlap(&mut self, other: &mut Self) -> usize {
        assert_eq!(self.len(), other.len());
        let mut ovlp = 0;
        for (&a, &b) in self.dense().iter().zip(other.dense()) {
            if a > 0 && b > 0 {
                ovlp += 1;
            }
        }
        ovlp
    }
}

#[cfg(test)]
mod tests {
    use crate::SDR;

    #[test]
    fn basic() {
        let mut a = SDR::new(111);
        assert_eq!(a.len(), 111);
        assert_eq!(a.sparsity(), 0.0);

        let mut b = SDR::ones(33);
        assert_eq!(b.len(), 33);
        assert_eq!(b.sparsity(), 1.0);

        let z = SDR::new(0);
        let z = SDR::ones(0);
    }

    #[test]
    fn convert() {
        let mut a = SDR::from_dense(vec![0, 1, 0, 2, 0, 11].into());
        assert_eq!(a.sparse().0, [1, 3, 5]);
        assert_eq!(a.sparse().1, [1, 2, 11]);

        let mut b = SDR::new(3);
        assert_eq!(b.sparse().0, &[]);
        assert_eq!(b.dense().to_vec(), [0, 0, 0]);

        let mut c = SDR::from_sparse(6, vec![1, 3, 5], Some(vec![1, 2, 11]));
        assert_eq!(c.dense(), a.dense());
    }

    #[test]
    fn overlap() {
        let mut a = SDR::from_sparse(100, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], None);
        let mut b = SDR::from_sparse(100, vec![5, 6, 7, 8, 9, 10, 11, 12, 13, 14], None);
        assert_eq!(a.overlap(&mut b), 5);
    }

    #[test]
    fn random() {
        let mut a = SDR::random(100, 0.1);
        assert_eq!(a.len(), 100);
        assert_eq!(a.sparsity(), 0.1);

        let mut b = SDR::random(100, 0.1);
        assert!(a.dense() != b.dense());
    }
}
