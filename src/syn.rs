use crate::{Idx, SDR};
use rand::prelude::*;

pub struct Synapses {
    num_axons: usize,
    num_dendrites: usize,

    dendrites: Vec<Vec<Synapse>>,
    dendrite_num_connected: Vec<usize>,

    axon_targets: Vec<Vec<usize>>,

    synapse_type_: SynapseType,
}

#[derive(Copy, Clone, Debug)]
pub enum SynapseType {
    Proximal,
    Distal,
}

#[derive(Clone)]
struct Synapse {
    axon: usize,
    permanence: f32,
}

impl Synapses {
    pub fn new(synapse_type: SynapseType) -> Self {
        Self {
            num_axons: 0,
            num_dendrites: 0,
            synapse_type_: synapse_type,
            dendrites: vec![],
            dendrite_num_connected: vec![],
            axon_targets: vec![],
        }
    }

    pub fn axons(&self) -> usize {
        self.num_axons
    }

    pub fn dendrites(&self) -> usize {
        self.num_dendrites
    }

    pub fn add_axons(&mut self, num_axons: usize) -> std::ops::Range<usize> {
        let start_range = self.num_axons;
        self.num_axons += num_axons;
        start_range..self.num_axons
    }

    pub fn add_dendrites(&mut self, num_dendrites: usize) -> std::ops::Range<usize> {
        let start_range = self.num_dendrites;
        self.num_dendrites += num_dendrites;
        start_range..self.num_dendrites
    }

    /// Randomly sample the active axons.
    pub fn grow(&mut self, activity: &mut SDR, weights: &[f32], dendrite: usize) {
        let mut rng = rand::thread_rng();
        let index = activity.sparse();
        let num_grow = index.len().min(weights.len());
        let sample = index.choose_multiple(&mut rng, num_grow);

        for (i, &x) in sample.enumerate() {
            self.dendrites[dendrite].push(Synapse {
                axon: x,
                permanence: weights[i],
            });
        }
        match self.synapse_type_ {
            SynapseType::Proximal => {}
            SynapseType::Distal => {}
        }
    }

    pub fn activate(&self, activity: &mut SDR) -> ndarray::Array1<f32> {
        let mut accumulators = vec![0.0; self.dendrites()];

        for &axon in activity.sparse().iter() {
            for &dendrite in &self.axon_targets[axon] {
                accumulators[dendrite] += 1.0;
            }
        }
        accumulators.into()
    }

    pub fn learn(&mut self, presyn: &mut SDR, postsyn: &mut SDR) {}

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use crate::SDR;

    #[test]
    fn basic() {
        todo!()
    }
}
