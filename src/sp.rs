use crate::{SynapseType, Synapses, SDR};

pub struct Parameters {
    pub num_cells: usize,
    pub sparsity: f32,
    pub threshold: f32,
    pub potential_percent: f32,
    pub num_dendrites: usize,
    pub permanence_increment: f32,
    pub permanence_decrement: f32,
    pub homeostatic_period: f32,
    pub stability_period: f32,
    pub fatigue_period: f32,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            num_cells: 2000,
            sparsity: 0.02,
            threshold: 1.0,
            potential_percent: 0.5,
            num_dendrites: 1,
            permanence_increment: 0.02,
            permanence_decrement: 0.005,
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

    /// New SP with no topology.
    pub fn new(parameters: Parameters) -> Self {
        let args = parameters;
        let syn = Synapses::new(SynapseType::Proximal);
        let af = vec![1.0 / args.sparsity; args.num_cells];
        Self { args, syn, af }
    }

    fn lazy_init(&mut self, inputs: &mut SDR) {
        if self.syn.axons() > 0 {
            return;
        }
        self.syn.add_axons(inputs.num_cells());
        self.syn
            .add_dendrites(self.args.num_cells * self.args.num_dendrites);
        let mut all_inputs = SDR::ones(inputs.num_cells());
        for d in 0..self.syn.dendrites() {
            self.syn.grow(&mut all_inputs, &[0.5], d);
        }
    }

    pub fn reset(&mut self) {
        self.syn.reset();
    }

    pub fn activate(&mut self, inputs: &mut SDR, learn: bool) -> SDR {
        self.lazy_init(inputs);

        let activity = todo!();

        if learn {
            self.learn(inputs, &mut activity)
        };
        activity
    }

    pub fn learn(&mut self, inputs: &mut SDR, activity: &mut SDR) {
        self.lazy_init(inputs);
        todo!()
    }
}
