use crate::{Idx, Synapses, SDR};

pub struct Parameters {
    pub num_cells: usize,
    pub sparsity: f32,
    pub threshold: f32,
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
            num_cells: 2048,
            sparsity: 0.02,
            threshold: 1.0,
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
        let af = vec![1.0 / args.sparsity; args.num_cells];
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

        // Normalize the activity by the number of connected synapses.
        // todo!();

        // Apply homeostatic boosting based on the cell activation frequency.
        // todo!();

        // Run the Winner-Takes-All Competition.
        let num_active = (self.args.sparsity * self.args.num_cells as f32).round() as usize;
        let mut sparse: Vec<_> = (0..self.args.num_cells as Idx).collect();
        sparse.select_nth_unstable_by_key(num_active, |&x| u32::MAX - connected[x as usize]);
        sparse.resize(num_active, 0);

        // Apply the activation threshold.
        // todo!();

        if learn {
            if sparse.len() < num_active {
                // Do an argpart of the AF to find the cells to use.
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut sp = SpatialPooler::new(Parameters::default());
        let mut inp1 = SDR::random(500, 0.1);
        let mut inp2 = SDR::random(500, 0.1);
        let mut a = sp.advance(&mut inp1, true);
        dbg!(&sp.syn);
        dbg!(&a);
        sp.reset();
        let mut b = sp.advance(&mut inp1, true);
        let mut c = sp.advance(&mut inp2, true);
        assert!(a.sparse() == b.sparse());
        assert!(a.sparse() != c.sparse());
    }
}
