use crate::{Idx, Synapses, SDR};

pub struct Parameters {
    pub num_cells: usize,
    pub sparsity: f32,
    pub threshold: usize,
    pub potential_percent: f32,
    pub num_dendrites: usize,
    pub learning_period: f32,
    pub coincidence_ratio: f32,
    pub homeostatic_period: f32,
    pub stability_period: f32,
    pub fatigue_period: f32,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            num_cells: 2000,
            sparsity: 0.02,
            threshold: 10,
            potential_percent: 0.5,
            num_dendrites: 1,
            learning_period: 0.02,
            coincidence_ratio: 0.005,
            homeostatic_period: 1000.0,
            stability_period: 0.0,
            fatigue_period: 0.0,
        }
    }
}

pub struct SpatialPooler {
    args: Parameters,
    syn: Synapses,
    af: Vec<f32>,
    // Stability & fatigue state vectors.
}

impl SpatialPooler {
    pub fn parameters(&self) -> &Parameters {
        &self.args
    }

    pub fn synapses(&self) -> &Synapses {
        &self.syn
    }

    pub fn new(parameters: Parameters) -> Self {
        let args = parameters;
        let mut syn = Synapses::default();
        syn.add_dendrites(args.num_cells * args.num_dendrites);
        let af = vec![args.sparsity; args.num_cells];
        Self { args, syn, af }
    }

    fn lazy_init(&mut self, inputs: &mut SDR) {
        if self.syn.num_axons() > 0 {
            return;
        }
        self.syn.add_axons(inputs.num_cells());
    }

    pub fn reset(&mut self) {
        self.syn.reset();
    }

    pub fn advance(&mut self, inputs: &mut SDR, learn: bool) -> SDR {
        self.lazy_init(inputs);

        let (connected, potential) = self.syn.activate(inputs);

        // Process the active synapse inputs into a general "activity"
        let boost_factor_adjust = 1.0 / self.args.sparsity.log2();
        let mut activity: Vec<_> = connected
            .iter()
            .zip(&self.syn.dend_num_connected_)
            .zip(&self.af)
            .map(|((&x, &nsyn), &af)| {
                if nsyn == 0 {
                    f32::INFINITY
                } else {
                    // Normalize the activity by the number of connected synapses.
                    let x = (x as f32) / (nsyn as f32);
                    // Apply homeostatic control based on the cell activation frequency.
                    x * af.log2() * boost_factor_adjust
                }
            })
            .collect();

        // Run the Winner-Takes-All Competition.
        let num_active = (self.args.sparsity * self.args.num_cells as f32).round() as usize;
        let mut sparse: Vec<_> = (0..self.args.num_cells as Idx).collect();
        sparse.select_nth_unstable_by(num_active, |&x, &y| {
            cmp_f32(activity[x as usize], activity[y as usize]).reverse()
        });
        sparse.resize(num_active, 0);

        // Apply the activation threshold.
        let mut sparse: Vec<_> = sparse
            .into_iter()
            .filter(|&cell| connected[cell as usize] > self.args.threshold as Idx)
            .collect();

        // Assign new cells to activate.
        if learn && sparse.len() < num_active {
            let num_new = num_active - sparse.len();
            // Select the cells with the lowest activation frequency.
            let mut min_af: Vec<_> = (0..self.args.num_cells as Idx).collect();
            min_af.select_nth_unstable_by(num_new, |&x, &y| {
                cmp_f32(self.af[x as usize], self.af[y as usize])
            });
            min_af.resize(num_new, 0);
            sparse.append(&mut min_af);
        }

        let mut activity = SDR::from_sparse(self.args.num_cells, sparse);

        if learn {
            self.learn(inputs, &mut activity);
        };
        return activity;
    }

    pub fn learn(&mut self, inputs: &mut SDR, activity: &mut SDR) {
        self.lazy_init(inputs);
        self.update_af(activity);
        //
        let incr = 1.0 - (-1.0 / self.args.learning_period).exp();
        let decr = -incr / self.args.coincidence_ratio;
        self.syn.hebbian(inputs, activity, incr, decr);
        // Grow new synapses.
        let num_syn = (inputs.num_cells() as f32 * self.args.potential_percent).round() as usize;
        for &dend in activity.sparse() {
            let weights = vec![0.5; num_syn];
            // TODO: Randomize the synapse weights.

            // TODO: How to implement the potential pool?
            // Or some other way of limiting the synapse growth?
            //     Potential pool is easy to impl: hash(axon & dend) < potential_percent
            self.syn.grow_synapses(inputs, dend, &weights);
        }
    }

    fn update_af(&mut self, activity: &mut SDR) {
        let alpha = (-1.0f32 / self.args.homeostatic_period).exp();
        let beta = 1.0 - alpha;
        for x in self.af.iter_mut() {
            *x *= alpha;
        }
        for i in activity.sparse() {
            self.af[*i as usize] += beta;
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
        let mut sp = SpatialPooler::new(Parameters::default());
        // Start with some random initial synapses.
        for _ in 0..100 {
            sp.advance(&mut SDR::random(500, 0.1), true);
        }
        sp.reset();
        dbg!(&sp.syn);
        //
        let mut inp1 = SDR::random(500, 0.1);
        let mut a = sp.advance(&mut inp1, true);

        // Test that similar SDRs yield similar outputs.
        for _trial in 0..10 {
            let mut inp2 = inp1.corrupt(0.2);
            let mut b = sp.advance(&mut inp1, true);
            assert!(a.percent_overlap(&mut b) > 0.5);
        }

        // Test that dissimilar SDRs yeild dissimilar outputs.
        for _trial in 0..10 {
            let mut inp3 = SDR::random(500, 0.1);
            let mut c = sp.advance(&mut inp3, true);
            assert!(a.percent_overlap(&mut c) < 0.5);
        }
    }
}
