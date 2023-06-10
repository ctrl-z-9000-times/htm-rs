use crate::{Idx, Synapses, SDR};
use pyo3::prelude::*;

#[pyclass]
pub struct SpatialPooler {
    num_cells: usize,
    num_active: usize,
    active_thresh: usize,
    potential_pct: f32,
    learning_period: f32,
    incidence_rate: f32,
    homeostatic_period: Option<f32>,

    pub syn: Synapses,
    af: Vec<f32>,

    num_steps: usize,
    step: usize,
    buffer: Vec<SDR>,
}

#[pymethods]
impl SpatialPooler {
    #[new]
    // #[args()] // TODO: Make the python version pretty w/ default values.
    pub fn new(
        num_cells: usize,
        num_active: usize,
        active_thresh: usize,
        potential_pct: f32,
        learning_period: f32,
        incidence_rate: f32,
        homeostatic_period: Option<f32>,
        num_steps: usize,
    ) -> Self {
        let mut syn = Synapses::new(incidence_rate, None);
        syn.add_dendrites(num_cells);
        let sparsity = num_active as f32 / num_cells as f32;
        return Self {
            num_cells,
            num_active,
            active_thresh,
            potential_pct,
            learning_period,
            incidence_rate,
            homeostatic_period,
            syn,
            af: if homeostatic_period.is_some() {
                vec![sparsity; num_cells]
            } else {
                vec![]
            },
            num_steps,
            step: 0,
            buffer: vec![SDR::zeros(0); num_steps],
        };
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

    fn __str__(&self) -> String {
        return format!("{}", self);
    }

    pub fn reset(&mut self) {
        self.buffer = vec![SDR::zeros(0); self.num_steps()];
        self.syn.reset();
        self.syn.clean();
    }

    pub fn advance(&mut self, inputs: &mut SDR, learn: bool, output: Option<&mut SDR>) -> SDR {
        assert!(output.is_none() || learn);
        self.lazy_init(inputs);

        let (connected, potential) = self.syn.activate(inputs);

        // Normalize the activity by the number of connected synapses.
        let mut activity: Vec<_> = connected
            .iter()
            .zip(self.syn.get_num_connected())
            .map(|(&x, &nsyn)| if x == 0 { 0.0 } else { (x as f32) / (nsyn as f32) })
            .collect();

        // Apply homeostatic control based on the cell activation frequency.
        if self.homeostatic_period.is_some() {
            let sparsity = self.num_active as f32 / self.num_cells as f32;
            let boost_factor_adjust = 1.0 / sparsity.log2();
            for (x, f) in activity.iter_mut().zip(&self.af) {
                *x = *x * f.log2() * boost_factor_adjust;
            }
        }

        // Run the Winner-Takes-All Competition.
        let mut sparse = Self::competition(&activity, self.num_active);

        // Apply the activation threshold.
        sparse.retain(|&cell| connected[cell as usize] > self.active_thresh as Idx);

        // Assign new cells to activate.
        // Only when doing unsupervised learning.
        if learn && output.is_none() {
            self.activate_least_used(&mut sparse, self.num_active);
        }

        let mut activity = SDR::from_sparse(self.num_cells, sparse);

        if let Some(output) = output {
            // Learn the association: input[t-num_steps] -> output[t]
            if self.num_steps() > 0 {
                std::mem::swap(inputs, &mut self.buffer[self.step]);
                self.step = (self.step + 1) % self.num_steps(); // Rotate our index into the circular buffer.
                if inputs.num_cells() == 0 {
                    *inputs = SDR::zeros(self.num_inputs());
                }
            }

            self.syn.learn(inputs, output, self.learning_period);

            // Grow new synapses.
            for &dend in output.sparse() {
                self.syn
                    .grow_competitive(inputs, dend, self.potential_pct, || self.incidence_rate);
            }

            // Depress the synapses leading to the incorrect outputs.
            // let incorrect = active - output;
            // self.syn.hebbian(&mut input, &mut incorrect, decr, 0.0);
            //
        } else if learn {
            self.update_af(&mut activity);

            self.syn.learn(inputs, &mut activity, self.learning_period);

            // Grow new synapses.
            for &dend in activity.sparse() {
                self.syn
                    .grow_competitive(inputs, dend, self.potential_pct, || self.incidence_rate);
            }
        };
        return activity;
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
        if sparse.len() < num_active {
            let num_new = num_active - sparse.len();
            // Select the cells with the lowest activation frequency.
            let mut min_af: Vec<_> = (0..self.num_cells as Idx).collect();
            min_af.select_nth_unstable_by(num_new, |&x, &y| cmp_f32(self.af[x as usize], self.af[y as usize]));
            min_af.resize(num_new, 0);
            sparse.append(&mut min_af);
        }
    }

    fn update_af(&mut self, activity: &mut SDR) {
        if let Some(period) = self.homeostatic_period {
            let decay = (-1.0f32 / period).exp();
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
        panic!()
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

    #[test]
    fn basic() {
        let make_sdr = || SDR::random(1000, 0.1);
        let mut sp = SpatialPooler::new(
            2000,        // num_cells
            40,          // num_active
            10,          // active_thresh
            0.5,         // potential_pct
            10.0,        // learning_period
            0.1,         // incidence_rate
            Some(100.0), // homeostatic_period
            0,
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
        let num_cells = 2000;
        let num_active = 40;
        let input_sdr = || SDR::random(100_000, 0.001);
        let output_sdr = || SDR::random(num_cells, num_active as f32 / num_cells as f32);
        let mut input_seq: Vec<SDR> = (0..seq_len).map(|_| input_sdr()).collect();
        let mut output_seq: Vec<SDR> = (0..seq_len).map(|_| output_sdr()).collect();

        let mut nn = SpatialPooler::new(
            num_cells,  // num_cells
            num_active, // num_active
            10,         // active_thresh
            0.3,        // potential_pct
            10.0,       // learning_period
            0.01,       // incidence_rate
            None,       // homeostatic_period
            delay,      // num_steps
        );

        // Train.
        for trial in 0..10 {
            for t in 0..seq_len {
                nn.advance(&mut input_seq[t].clone(), true, Some(&mut output_seq[t].clone()));
            }
        }

        // This reset prevents the last elements of the training sequence from
        // learning the random noise.
        nn.reset();

        for noise in 0..3 * seq_len {
            nn.advance(&mut input_sdr(), true, Some(&mut output_sdr()));
        }

        // Test.
        nn.syn.clean();
        println!("{}", &nn);

        for t in 0..seq_len {
            let mut prediction = nn.advance(&mut input_seq[t].clone(), false, None);
            let correct = &mut output_seq[(t + delay) % seq_len];
            let mut overlap = prediction.percent_overlap(correct);
            dbg!(overlap);
            assert!(overlap >= 0.90);
        }
    }
}
