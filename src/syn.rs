use crate::{Idx as CellIdx, SDR};
use rand::prelude::*;

pub type SynIdx = u32;
pub type DendIdx = u32;

#[derive(Default)]
pub struct Synapses {
    num_axons_: usize,
    num_dendrites_: usize,
    num_synapses_: usize,

    // Parallel arrays of synapse data.
    syn_axons_: Vec<CellIdx>,
    syn_dendrites_: Vec<DendIdx>,
    syn_permanences_: Vec<f32>,

    /// The synapses are sorted by dendrite, which is indexed here.
    /// If this is `None` then the synapses are not sorted.
    dend_ranges_: Option<Vec<std::ops::Range<SynIdx>>>,

    dend_num_connected_: Vec<usize>,

    /// Each axon has a list of target dendrites, which is partitioned into two
    /// sections for potential and connected synapses.
    ///
    /// let partition = axon_targets_[axons].0
    /// let potential = axon_targets_[axons].1[ 0 .. partition ]
    /// let connected = axon_targets_[axons].1[ partition .. ]
    axon_targets_: Vec<(SynIdx, Vec<DendIdx>)>,
}

impl Synapses {
    pub fn num_axons(&self) -> usize {
        self.num_axons_
    }

    pub fn num_dendrites(&self) -> usize {
        self.num_dendrites_
    }

    pub fn num_synapses(&self) -> usize {
        self.num_synapses_
    }

    pub fn add_axons(&mut self, num_axons: usize) -> std::ops::Range<usize> {
        let start_range = self.num_axons_;
        self.num_axons_ += num_axons;
        self.axon_targets_.resize(self.num_axons_, (0, vec![]));
        start_range..self.num_axons_
    }

    pub fn add_dendrites(&mut self, num_dendrites: usize) -> std::ops::Range<usize> {
        let start_range = self.num_dendrites_;
        self.num_dendrites_ += num_dendrites;
        self.dend_num_connected_.resize(self.num_dendrites_, 0);
        let num_synapses = self.num_synapses();
        if let Some(dend_ranges_) = &mut self.dend_ranges_ {
            dend_ranges_.resize(
                self.num_dendrites_,
                num_synapses as u32..num_synapses as u32,
            );
        }
        return start_range..self.num_dendrites_;
    }

    /// Randomly sample the active axons.
    pub fn add_synapses(&mut self, axons: &mut SDR, dendrite: DendIdx, weights: &[f32]) {
        let mut rng = rand::thread_rng();
        let index = axons.sparse();
        let num_grow = weights.len().min(index.len());
        let sample = index.choose_multiple(&mut rng, num_grow);

        for (&axon, &weight) in sample.zip(weights) {
            self.syn_axons_.push(axon);
            self.syn_dendrites_.push(dendrite);
            self.syn_permanences_.push(weight);
            self.num_synapses_ += 1;
            let axon_targets = &mut self.axon_targets_[axon as usize];
            if weight >= 0.5 {
                self.dend_num_connected_[dendrite as usize] += 1;
                axon_targets.1.push(dendrite);
            } else {
                axon_targets.1.insert(axon_targets.0 as usize, dendrite);
                axon_targets.0 += 1;
            }
        }
        self.dend_ranges_ = None;
    }

    pub fn activate(&self, activity: &mut SDR) -> (Vec<u32>, Vec<u32>) {
        let mut potential = vec![0; self.num_dendrites_];
        let mut connected = vec![0; self.num_dendrites_];

        for &axon in activity.sparse().iter() {
            let partition = self.axon_targets_[axon as usize].0 as usize;
            let targets = &self.axon_targets_[axon as usize].1;
            for &dendrite in &targets[0..partition] {
                potential[dendrite as usize] += 1;
            }
            for &dendrite in &targets[partition..] {
                potential[dendrite as usize] += 1;
                connected[dendrite as usize] += 1;
            }
        }
        (potential, connected)
    }

    pub fn learn(&mut self, presyn: &mut SDR, postsyn: &mut SDR) {
        // Hebbian learning
        todo!();

        // Grow new synapses as needed.
        todo!();
    }

    pub fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut x = Synapses::default();
        assert_eq!(x.add_axons(10), 0..10);
        assert_eq!(x.add_dendrites(10), 0..10);
        x.add_synapses(&mut SDR::ones(10), 3, &[0.6, 0.4, 0.5]);
        assert_eq!(x.num_synapses(), 3);
        let (pot, con) = x.activate(&mut SDR::ones(10));
        assert_eq!(pot, [0, 0, 0, 3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(con, [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]);
    }
}
