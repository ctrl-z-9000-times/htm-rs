use crate::{Idx, Synapses, SDR};
use pyo3::prelude::*;

#[pyclass]
pub struct SpatialPooler {
    num_cells: usize,
    num_active: usize,
    active_thresh: usize,
    potential_pct: f32,
    learning_period: f32,
    coincidence_ratio: f32,
    homeostatic_period: f32,

    syn: Synapses,
    af: Vec<f32>,
}

#[pymethods]
impl SpatialPooler {
    #[new]
    // #[args()] // TODO: Make the python version pretty w/ default values.
    fn new(
        num_cells: usize,
        num_active: usize,
        active_thresh: usize,
        potential_pct: f32,
        learning_period: f32,
        coincidence_ratio: f32,
        homeostatic_period: f32,
    ) -> Self {
        let mut syn = Synapses::new();
        syn.add_dendrites(num_cells);
        let sparsity = num_active as f32 / num_cells as f32;
        return Self {
            num_cells,
            num_active,
            active_thresh,
            potential_pct,
            learning_period,
            coincidence_ratio,
            homeostatic_period,
            syn,
            af: vec![sparsity; num_cells],
        };
    }

    fn __str__(&self) -> String {
        return format!(
            "SpatialPooler {{ num_cells: {}, num_active {} }}\n    {:?}",
            self.num_cells, self.num_active, self.syn
        );
    }

    pub fn reset(&mut self) {
        self.syn.reset();
    }

    pub fn advance(&mut self, inputs: &mut SDR, learn: bool) -> SDR {
        self.lazy_init(inputs);

        let (connected, potential) = self.syn.activate(inputs);

        // Process the active synapse inputs into a general "activity"
        let sparsity = self.num_active as f32 / self.num_cells as f32;
        let boost_factor_adjust = 1.0 / sparsity.log2();
        let mut activity: Vec<_> = connected
            .iter()
            .zip(self.syn.get_num_connected())
            .zip(&self.af)
            .map(|((&x, &nsyn), &af)| {
                if nsyn == 0 {
                    0.0 // Zero div Zero is Zero.
                } else {
                    // Normalize the activity by the number of connected synapses.
                    let x = (x as f32) / (nsyn as f32);
                    // Apply homeostatic control based on the cell activation frequency.
                    let x = x * af.log2() * boost_factor_adjust;
                    x
                }
            })
            .collect();

        // Run the Winner-Takes-All Competition.
        let mut sparse: Vec<_> = (0..self.num_cells as Idx).collect();
        sparse.select_nth_unstable_by(self.num_active, |&x, &y| {
            cmp_f32(activity[x as usize], activity[y as usize]).reverse()
        });
        sparse.resize(self.num_active, 0);

        // Apply the activation threshold.
        let mut sparse: Vec<_> = sparse
            .into_iter()
            .filter(|&cell| connected[cell as usize] > self.active_thresh as Idx)
            .collect();

        // Assign new cells to activate.
        if learn && sparse.len() < self.num_active {
            let num_new = self.num_active - sparse.len();
            // Select the cells with the lowest activation frequency.
            let mut min_af: Vec<_> = (0..self.num_cells as Idx).collect();
            min_af.select_nth_unstable_by(num_new, |&x, &y| {
                cmp_f32(self.af[x as usize], self.af[y as usize])
            });
            min_af.resize(num_new, 0);
            sparse.append(&mut min_af);
        }

        let mut activity = SDR::from_sparse(self.num_cells, sparse);

        if learn {
            self.learn(inputs, &mut activity);
        };
        return activity;
    }

    pub fn learn(&mut self, inputs: &mut SDR, activity: &mut SDR) {
        self.lazy_init(inputs);
        self.update_af(activity);
        // Hebbian Learning.
        let incr = 1.0 - (-1.0 / self.learning_period).exp();
        let decr = -incr / self.coincidence_ratio;
        self.syn.hebbian(inputs, activity, incr, decr);
        // Grow new synapses.
        for &dend in activity.sparse() {
            self.syn
                .grow_competitive(inputs, dend, self.potential_pct, || 0.5);
        }
    }
}

impl SpatialPooler {
    fn lazy_init(&mut self, inputs: &mut SDR) {
        if self.syn.num_axons() == 0 {
            self.syn.add_axons(inputs.num_cells());
        }
    }

    fn update_af(&mut self, activity: &mut SDR) {
        let alpha = (-1.0f32 / self.homeostatic_period).exp();
        let beta = 1.0 - alpha;
        for i in activity.sparse() {
            self.af[*i as usize] += beta;
        }
        for x in self.af.iter_mut() {
            *x *= alpha;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let make_sdr = || SDR::random(1000, 0.1);
        let mut sp = SpatialPooler::new(
            2000,  // num_cells
            40,    // num_active
            10,    // active_thresh
            0.5,   // potential_pct
            10.0,  // learning_period
            10.0,  // coincidence_ratio
            100.0, // homeostatic_period
        );
        sp.reset();
        dbg!(&sp.syn);
        //
        let mut inp1 = make_sdr();
        for _train in 0..10 {
            sp.advance(&mut inp1, true);
        }
        let mut a1 = sp.advance(&mut inp1, true);
        let mut a = sp.advance(&mut inp1, true);
        dbg!(a1.percent_overlap(&mut a));
        assert_eq!(a1.sparse(), a.sparse());

        // Test that similar SDRs yield similar outputs.
        for _trial in 0..10 {
            let mut inp2 = inp1.corrupt(0.2);
            let mut b = sp.advance(&mut inp1, true);
            assert!(a.percent_overlap(&mut b) > 0.5);
        }

        // Test that dissimilar SDRs yeild dissimilar outputs.
        for _trial in 0..10 {
            let mut inp3 = make_sdr();
            let mut c = sp.advance(&mut inp3, true);
            assert!(a.percent_overlap(&mut c) < 0.5);
        }
    }
}
