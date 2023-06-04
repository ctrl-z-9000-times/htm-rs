use crate::{Idx, SDR};
use rand::prelude::*;
use std::ops::Range;

#[derive(Default)]
pub struct Synapses {
    // Parallel arrays of synapse data.
    syn_axons_: Vec<Idx>,
    syn_dendrites_: Vec<Idx>,
    syn_permanences_: Vec<f32>,
    syn_sorted_: bool,

    // Parallel arrays of axon data.
    axon_syn_ranges_: Vec<Range<Idx>>,

    // Parallel arrays of dendrite data.
    dend_synapses_: Vec<Vec<Idx>>,
    pub dend_num_connected_: Vec<Idx>,
}

impl Synapses {
    pub fn num_axons(&self) -> usize {
        return self.axon_syn_ranges_.len();
    }

    pub fn num_dendrites(&self) -> usize {
        return self.dend_synapses_.len();
    }

    pub fn num_synapses(&self) -> usize {
        return self.syn_axons_.len();
    }

    pub fn add_axons(&mut self, num_axons: usize) -> Range<usize> {
        let start_range = self.num_axons();
        let end_range = start_range + num_axons;
        self.syn_sorted_ = false;
        //
        self.axon_syn_ranges_.resize(
            end_range,
            self.num_synapses() as Idx..self.num_synapses() as Idx,
        );
        return start_range..end_range;
    }

    pub fn add_dendrites(&mut self, num_dendrites: usize) -> Range<usize> {
        let start_range = self.num_dendrites();
        let end_range = start_range + num_dendrites;
        //
        self.dend_num_connected_.resize(end_range, 0 as Idx);
        self.dend_synapses_.resize(end_range, vec![]);
        return start_range..end_range;
    }

    pub fn add_synapse(&mut self, axon: Idx, dendrite: Idx, weight: f32) {
        self.syn_axons_.push(axon);
        self.syn_dendrites_.push(dendrite);
        self.syn_permanences_.push(weight);
        self.syn_sorted_ = false;
        if weight >= 0.5 {
            self.dend_num_connected_[dendrite as usize] += 1;
        }
    }

    /// Randomly sample the active axons.
    pub fn grow_synapses(&mut self, axons: &mut SDR, dendrite: Idx, weights: &[f32]) {
        debug_assert!(axons.num_cells() == self.num_axons());
        debug_assert!(dendrite < self.num_dendrites() as Idx);
        let mut rng = rand::thread_rng();
        let index = axons.sparse();
        let num_grow = weights.len().min(index.len());
        let sample = index.choose_multiple(&mut rng, num_grow);

        for (&axon, &weight) in sample.zip(weights) {
            self.add_synapse(axon, dendrite, weight);
        }
    }

    fn sort_by_axon(&mut self) {
        if !self.syn_sorted_ && self.num_synapses() > 0 {
            // Sort the synapses by their presynaptic axon.
            let mut argsort: Vec<_> = (0..self.num_synapses()).collect();
            argsort.sort_by_key(|&idx| self.syn_axons_[idx as usize]);
            self.syn_axons_ = argsort
                .iter()
                .map(|&x| self.syn_axons_[x as usize])
                .collect();
            self.syn_dendrites_ = argsort
                .iter()
                .map(|&x| self.syn_dendrites_[x as usize])
                .collect();
            self.syn_permanences_ = argsort
                .iter()
                .map(|&x| self.syn_permanences_[x as usize])
                .collect();
            // Scan for contiguous ranges of synaspes from the same axon.
            let num_axons = self.num_axons();
            self.axon_syn_ranges_ = Vec::with_capacity(num_axons);
            let mut start_syn: Idx = 0;
            let mut cur_axon = self.syn_axons_[0] as Idx;
            for syn in 1..self.num_synapses() as Idx {
                let axon = self.syn_axons_[syn as usize];
                if axon != cur_axon {
                    // Terminate this contiguous stretch.
                    self.axon_syn_ranges_.push(start_syn..syn);
                    start_syn = syn;
                    cur_axon = axon;
                    while (self.axon_syn_ranges_.len() as Idx) < axon {
                        self.axon_syn_ranges_.push(start_syn..start_syn);
                    }
                }
            }
            let num_syn = self.num_synapses() as Idx;
            self.axon_syn_ranges_.push(start_syn..num_syn); // Terminate the final range of synapses.
            self.axon_syn_ranges_.resize(num_axons, num_syn..num_syn); // Fill in any axons without synapases.

            // Update the dendrite synapses.
            self.dend_synapses_ = vec![vec![]; self.num_dendrites()];
            for syn in 0..self.num_synapses() as Idx {
                let dend = self.syn_dendrites_[syn as usize];
                self.dend_synapses_[dend as usize].push(syn);
            }
        }
    }

    pub fn activate(&mut self, activity: &mut SDR) -> (Vec<u32>, Vec<u32>) {
        assert_eq!(activity.num_cells(), self.num_axons());

        self.sort_by_axon();

        let mut potential = vec![0; self.num_dendrites()];
        let mut connected = vec![0; self.num_dendrites()];

        for &axon in activity.sparse().iter() {
            for syn in self.axon_syn_ranges_[axon as usize].clone() {
                let dend = self.syn_dendrites_[syn as usize];
                let perm = self.syn_permanences_[syn as usize];
                potential[dend as usize] += 1;
                if perm >= 0.5 {
                    connected[dend as usize] += 1;
                }
            }
        }
        return (potential, connected);
    }

    pub fn hebbian(&mut self, presyn: &mut SDR, postsyn: &mut SDR, incr: f32, decr: f32) {
        let presyn = presyn.dense();
        self.sort_by_axon();
        for &dend in postsyn.sparse() {
            for syn in self.dend_synapses_[dend as usize].clone() {
                let axon = self.syn_axons_[syn as usize];
                let presyn_active = presyn[axon as usize];
                let permanence = &mut self.syn_permanences_[syn as usize];
                let old_state = *permanence > 0.5;
                *permanence += if presyn_active { incr } else { decr };
                let new_state = *permanence > 0.5;
                if old_state != new_state {
                    if new_state {
                        self.dend_num_connected_[dend as usize] += 1;
                    } else {
                        self.dend_num_connected_[dend as usize] -= 1;
                    }
                }
            }
        }
    }

    pub fn reset(&mut self) {}
}

impl std::fmt::Debug for Synapses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Synapses (Axons: {}, Dendrites: {} Synapses: {})\n",
            self.num_axons(),
            self.num_dendrites(),
            self.num_synapses(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut x = Synapses::default();
        assert_eq!(x.add_axons(10), 0..10);
        assert_eq!(x.add_dendrites(10), 0..10);
        x.grow_synapses(&mut SDR::ones(10), 3, &[0.6, 0.4, 0.5]);
        assert_eq!(x.num_synapses(), 3);
        let (pot, con) = x.activate(&mut SDR::ones(10));
        assert_eq!(pot, [0, 0, 0, 3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(con, [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn hebbian() {
        let mut x = Synapses::default();
        x.add_axons(2);
        x.add_dendrites(1);
        x.grow_synapses(&mut SDR::ones(2), 0, &[0.5, 0.5]);
        let mut pattern = SDR::from_dense(&[false, true]);
        x.hebbian(&mut pattern, &mut SDR::ones(1), 0.1, -0.1);
        let (pot, con) = x.activate(&mut SDR::ones(2));
        assert_eq!(pot, [2]);
        assert_eq!(con, [1]);
    }
}
