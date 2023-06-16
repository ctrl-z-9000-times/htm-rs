use crate::{Idx, Synapses, SDR};
use pyo3::prelude::*;

/// Spatial Pooler Algorithm
///
/// For more information see:  
/// > The HTM Spatial Pooler-A Neocortical Algorithm for Online Sparse Distributed Coding  
/// > Yuwei Cui, Subutai Ahmad and Jeff Hawkins (2017)  
/// > <https://doi.org/10.3389/fncom.2017.00111>
#[pyclass]
pub struct SpatialPooler {
    num_cells: usize,
    num_active: usize,
    num_steps: usize,
    threshold: f32,
    potential_pct: f32,
    learning_period: usize,
    boosting_period: Option<usize>,

    syn: Synapses,
    af: Vec<f32>,
    step: usize,
    buffer: Vec<SDR>,
}

#[pymethods]
impl SpatialPooler {
    /// Argument **num_cells**:  
    /// Argument **num_active**:  
    /// Argument **num_steps**: Number of time-steps into the future to predict.  
    /// Argument **threshold**: Fraction of the input activity which cells must match before they can activate.  
    /// Argument **potential_pct**: Fraction of the input axons which each cell can connect to.  
    /// Argument **learning_period**: Time constant of the exponential moving average that controls
    ///     the synaptic weights. This controls how fast it learns.
    /// Argument **num_patterns**:  Maximum number of different SDRs that dendrite can learn about
    ///     before it starts pruning off the least used synapses.  
    /// Argument **weight_gain**: Slope of the synapse weight function.
    ///     This controls how quickly hebbian learning saturates.
    ///     A value of 1.0 will have the maximum linear range, and larger values will divide the linear range.  
    /// Argument **boosting_period**:  
    /// Argument **seed**: For random number generation. If None is given then this uses a OS-random.  
    #[new]
    // #[pyo3(signature = (seed = "None"))]
    pub fn new(
        num_cells: usize,
        num_active: usize,
        num_steps: usize,
        threshold: f32,
        potential_pct: f32,
        learning_period: usize,
        num_patterns: usize,
        weight_gain: f32,
        boosting_period: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let mut syn = Synapses::new(num_patterns, weight_gain, seed);
        syn.add_dendrites(num_cells);
        return Self {
            num_cells,
            num_active,
            num_steps,
            threshold,
            potential_pct,
            learning_period,
            boosting_period,
            syn,
            af: if boosting_period.is_some() {
                let sparsity = num_active as f32 / num_cells as f32;
                vec![sparsity; num_cells]
            } else {
                vec![]
            },
            step: 0,
            buffer: vec![SDR::zeros(0); num_steps],
        };
    }

    /// Grow random synapses on every cell in the spatial pooler.
    pub fn initialize_synapses(&mut self, num_inputs: usize, num_synapses: usize) {
        let mut all_inputs = SDR::ones(num_inputs);
        self.lazy_init(&mut all_inputs);
        let pp = num_synapses as f32 / num_inputs as f32;
        for den in 0..self.syn.num_dendrites() {
            self.syn.grow_competitive(&mut all_inputs, den as Idx, pp);
        }
    }

    pub fn num_cells(&self) -> usize {
        return self.num_cells;
    }
    pub fn num_inputs(&self) -> usize {
        return self.syn.num_axons();
    }
    pub fn num_outputs(&self) -> usize {
        return self.num_cells();
    }
    pub fn num_steps(&self) -> usize {
        return self.num_steps;
    }
    pub fn seed(&self) -> u64 {
        return self.syn.seed();
    }

    fn __str__(&self) -> String {
        return format!("{}", self);
    }

    pub fn reset(&mut self) {
        self.buffer = vec![SDR::zeros(0); self.num_steps()];
        self.syn.reset();
        self.syn.clean();
    }

    pub fn advance(&mut self, mut inputs: &mut SDR, learn: bool, output: Option<&mut SDR>) -> SDR {
        self.lazy_init(inputs);

        let (potential, mut connected) = self.syn.activate(inputs);

        self.apply_boosting(&mut connected);

        // Supervised learning mode.
        if let Some(output) = output {
            // Run the Winner-Takes-All Competition.
            let mut sparse = Self::competition(&connected, self.num_active);

            // Apply the activation threshold.
            let threshold = self.threshold * self.potential_pct * inputs.num_active() as f32;
            sparse.retain(|&cell| connected[cell as usize] > threshold);

            let mut activity = SDR::from_sparse(self.num_cells, sparse);

            if learn {
                if self.num_steps() > 0 {
                    let mut prev_inputs = std::mem::replace(&mut self.buffer[self.step], inputs.clone());
                    self.step = (self.step + 1) % self.num_steps(); // Rotate our index into the circular buffer.
                    if prev_inputs.num_cells() == 0 {
                        prev_inputs = SDR::zeros(self.num_inputs());
                    }
                    self.learn(&mut prev_inputs, output);
                } else {
                    self.learn(inputs, output);
                }

                // Depress the synapses leading to the incorrect outputs.
                // let incorrect = active - output;
                // self.syn.hebbian(&mut input, &mut incorrect, decr, 0.0);
                //
            }
            return activity;
        }
        // Unsupervised learning mode.
        else {
            // Normalize the activity by the sum of synapse weights to the cell.
            let normalized: Vec<_> = connected
                .iter()
                .zip(self.syn.get_sum_weights())
                .map(|(x, w)| if *w > f32::EPSILON { *x / w } else { 0.0 })
                .collect();

            // Run the Winner-Takes-All Competition.
            let mut sparse = Self::competition(&normalized, self.num_active);

            // Apply the activation threshold.
            sparse.retain(|&cell| normalized[cell as usize] > self.threshold);

            // Assign new cells to activate.
            // Only when doing unsupervised learning.
            if learn {
                self.activate_least_used(&mut sparse, self.num_active);
            }

            let mut activity = SDR::from_sparse(self.num_cells, sparse);

            if learn {
                self.learn(inputs, &mut activity);
            }
            return activity;
        }
    }
}

impl SpatialPooler {
    fn lazy_init(&mut self, inputs: &mut SDR) {
        if self.syn.num_axons() == 0 {
            self.syn.add_axons(inputs.num_cells());
        } else {
            assert!(inputs.num_cells() == self.num_inputs());
        }
    }

    /// Run the Winner-Takes-All Competition.
    fn competition(activity: &[f32], num_active: usize) -> Vec<Idx> {
        let mut sparse: Vec<_> = (0..activity.len() as Idx).collect();
        sparse.select_nth_unstable_by(num_active, |&x, &y| {
            cmp_f32(activity[x as usize], activity[y as usize]).reverse()
        });
        sparse.resize(num_active, 0);
        return sparse;
    }

    fn activate_least_used(&self, sparse: &mut Vec<Idx>, num_active: usize) {
        if sparse.len() < num_active && self.boosting_period.is_some() {
            let num_new = num_active - sparse.len();
            // Select the cells with the lowest activation frequency.
            let mut min_af: Vec<_> = (0..self.num_cells as Idx).collect();
            min_af.select_nth_unstable_by(num_new, |&x, &y| cmp_f32(self.af[x as usize], self.af[y as usize]));
            min_af.resize(num_new, 0);
            sparse.append(&mut min_af);
        }
    }

    fn learn(&mut self, inputs: &mut SDR, activity: &mut SDR) {
        self.update_af(activity);

        // Hebbian learning.
        self.syn.learn(inputs, activity, self.learning_period as f32);

        // Grow new synapses.
        for &dend in activity.sparse() {
            self.syn.grow_competitive(inputs, dend, self.potential_pct);
        }
    }

    fn apply_boosting(&self, activity: &mut Vec<f32>) {
        // Apply homeostatic control based on the cell activation frequency.
        if self.boosting_period.is_some() {
            let sparsity = self.num_active as f32 / self.num_cells as f32;
            let boost_factor_adjust = 1.0 / sparsity.log2();
            for (x, f) in activity.iter_mut().zip(&self.af) {
                *x = *x * f.log2() * boost_factor_adjust;
            }
        }
    }

    fn update_af(&mut self, activity: &mut SDR) {
        if let Some(period) = self.boosting_period {
            let decay = (-1.0f32 / period as f32).exp();
            let alpha = 1.0 - decay;
            // dbg!(alpha, decay);
            for (frq, state) in self.af.iter_mut().zip(activity.dense().iter()) {
                *frq += alpha * (*state as usize as f32 - *frq);
            }
        }
    }
}

fn cmp_f32(a: f32, b: f32) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    } else if a > b {
        std::cmp::Ordering::Greater
    } else if a == b {
        std::cmp::Ordering::Equal
    } else {
        panic!("{} ?= {}", a, b)
    }
}

impl std::fmt::Display for SpatialPooler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Spatial Pooler {}", self.syn,);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Stats;

    #[test]
    fn basic() {
        let make_sdr = || SDR::random(1000, 0.1);
        let mut sp = SpatialPooler::new(
            2000,      // num_cells
            40,        // num_active
            0,         // num steps
            0.1,       // threshold
            0.5,       // potential_pct
            10,        // learning_period
            10,        // num_patterns
            5.0,       // gain
            Some(100), // boosting_period
            None,      // Seed
        );
        sp.reset();
        println!("{}", &sp);
        //
        let mut inp1 = make_sdr();
        for _train in 0..10 {
            sp.advance(&mut inp1, true, None);
        }
        let mut a1 = sp.advance(&mut inp1, true, None);
        let mut a = sp.advance(&mut inp1, true, None);
        dbg!(a1.percent_overlap(&mut a));
        assert_eq!(a1.sparse(), a.sparse());

        // Test that similar SDRs yield similar outputs.
        for _trial in 0..10 {
            let mut inp2 = inp1.corrupt(0.2);
            let mut b = sp.advance(&mut inp1, true, None);
            assert!(a.percent_overlap(&mut b) > 0.5);
        }

        // Test that dissimilar SDRs yeild dissimilar outputs.
        for _trial in 0..10 {
            let mut inp3 = make_sdr();
            let mut c = sp.advance(&mut inp3, true, None);
            dbg!(a.percent_overlap(&mut c));
            assert!(a.percent_overlap(&mut c) < 0.5);
        }
    }

    #[test]
    fn prediction() {
        // Make an artificial sequence of SDRs to demonstrate the supervised predictor capabilities.
        let seq_len = 10;
        let delay = 3;
        let num_cells = 10000;
        let num_active = 200;
        let make_sdr = || SDR::random(num_cells, num_active as f32 / num_cells as f32);
        let mut sequence: Vec<SDR> = (0..seq_len).map(|_| make_sdr()).collect();

        let mut nn = SpatialPooler::new(
            num_cells,  // num_cells
            num_active, // num_active
            delay,      // num_steps
            0.01,       // threshold
            0.5,        // potential_pct
            5,          // learning_period
            20,         // num_patterns
            2.5,        // incidence_gain
            None,       // boosting_period
            None,       // seed
        );
        let mut stats = Stats::new(100.0);

        // nn.initialize_synapses(num_cells, 40);

        // Train.
        for trial in 0..3 {
            for t in 0..seq_len {
                let mut x = nn.advance(&mut sequence[t].clone(), true, Some(&mut sequence[t]));
                stats.update(&mut x);
            }
        }

        // This reset prevents the last elements of the training sequence from
        // learning the random noise.
        // nn.reset();

        for noise in 0..10 {
            nn.advance(&mut make_sdr(), true, Some(&mut make_sdr()));
        }

        // Test.
        nn.syn.clean();
        println!("{}", &nn);
        println!("{}", &stats);

        for t in 0..seq_len {
            let mut prediction = nn.advance(&mut sequence[t].clone(), false, Some(&mut sequence[t]));
            let correct = &mut sequence[(t + delay) % seq_len];
            let mut overlap = prediction.percent_overlap(correct);
            dbg!(overlap);
            assert!(overlap >= 0.90);
        }
        // panic!("END OF TEST")
    }
}
