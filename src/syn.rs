use crate::{Idx, SDR};
use rand::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::Range;

// TODO: Make the connected threshold into an argument?

pub struct Synapses {
    clean_: bool,
    seed_: u64,

    num_syn_: Idx,
    num_axn_: Idx,
    num_den_: Idx,

    // The synapses are sorted by presynaptic axon and then postsynaptic dendrite.

    // Parallel arrays of synapse data.
    syn_axons_: Vec<Idx>,
    syn_dendrites_: Vec<Idx>,
    syn_permanences_: Vec<f32>,

    // Parallel arrays of axon data.
    axn_syn_ranges_: Vec<Range<Idx>>,

    // Parallel arrays of dendrite data.
    den_synapses_: Vec<Vec<Idx>>,
    den_num_connected_: Vec<Idx>,
}

impl Synapses {
    pub fn new() -> Self {
        return Self {
            clean_: false,
            seed_: 42,
            num_syn_: 0,
            num_axn_: 0,
            num_den_: 0,
            syn_axons_: vec![],
            syn_dendrites_: vec![],
            syn_permanences_: vec![],
            axn_syn_ranges_: vec![],
            den_synapses_: vec![],
            den_num_connected_: vec![],
        };
    }

    pub fn reset(&mut self) {}

    pub fn num_axons(&self) -> usize {
        return self.num_axn_ as usize;
    }

    pub fn num_dendrites(&self) -> usize {
        return self.num_den_ as usize;
    }

    pub fn num_synapses(&self) -> usize {
        return self.num_syn_ as usize;
    }

    pub fn add_axons(&mut self, num_axons: usize) -> Range<usize> {
        self.clean_ = false;
        let start_range = self.num_axn_;
        self.num_axn_ = (self.num_axn_ as usize + num_axons).try_into().unwrap();
        return start_range as usize..self.num_axn_ as usize;
    }

    pub fn add_dendrites(&mut self, num_dendrites: usize) -> Range<usize> {
        self.clean_ = false;
        let start_range = self.num_den_;
        self.num_den_ = (self.num_den_ as usize + num_dendrites).try_into().unwrap();
        return start_range as usize..self.num_den_ as usize;
    }

    pub fn add_synapse(&mut self, axon: Idx, dendrite: Idx, weight: f32) {
        debug_assert!(axon < self.num_axons() as Idx);
        debug_assert!(dendrite < self.num_dendrites() as Idx);
        self.clean_ = false;
        self.num_syn_ += 1;
        self.syn_axons_.push(axon);
        self.syn_dendrites_.push(dendrite);
        self.syn_permanences_.push(weight);
    }

    pub fn get_num_connected(&self) -> &[Idx] {
        debug_assert!(self.clean_);
        return self.den_num_connected_.as_slice();
    }

    /// Randomly sample the active axons.
    pub fn grow_selective<F>(
        &mut self,
        axons: &mut SDR,
        dendrite: Idx,
        max_grow: usize,
        max_total: usize,
        mut weights: F,
    ) where
        F: FnMut() -> f32,
    {
        let mut rng = rand::thread_rng();
        let synapses = &self.den_synapses_[dendrite as usize];
        let num_syn = synapses.len();
        let num_grow = max_grow.min(max_total - num_syn);
        let sample = axons.sparse().choose_multiple(&mut rng, num_grow);

        // Discard presynapses which are already connected to.
        // This should immediately select new presynapses to replace them.
        // todo!();

        for &axon in sample {
            self.add_synapse(axon, dendrite, weights());
        }
    }

    ///
    pub fn grow_competitive<F>(
        &mut self,
        axons: &mut SDR,
        dendrite: Idx,
        potential_pct: f32,
        mut weights: F,
    ) where
        F: FnMut() -> f32,
    {
        // Calculate the potential pool of presynapses for this dendrite.
        let estimated_pool_size = 2 * (potential_pct * axons.num_active() as f32).round() as usize;
        let mut potential_pool = Vec::with_capacity(estimated_pool_size);
        // Hash the dendrite-axon pair into a stable, unique, and psuedo-random number.
        // Use that number to decide which synapses are potentially connected.
        let mut hash_state = DefaultHasher::new();
        self.seed_.hash(&mut hash_state);
        dendrite.hash(&mut hash_state);
        for &axon in axons.sparse() {
            let mut syn_hash_state = hash_state.clone();
            axon.hash(&mut syn_hash_state);
            let syn_hash: u64 = syn_hash_state.finish();
            let syn_rnd_num = (syn_hash as f64) / (u64::MAX as f64);
            if syn_rnd_num < potential_pct as f64 {
                potential_pool.push(axon);
            }
        }
        // Make synapses to every axon in the potential pool. This will make
        // duplicate synapses, which will be removed next time it's cleaned.
        for axon in potential_pool {
            self.add_synapse(axon, dendrite, weights());
        }
    }

    pub fn activate(&mut self, activity: &mut SDR) -> (Vec<u32>, Vec<u32>) {
        debug_assert_eq!(activity.num_cells(), self.num_axons());
        self.clean();

        let mut potential = vec![0; self.num_dendrites()];
        let mut connected = vec![0; self.num_dendrites()];

        for &axon in activity.sparse().iter() {
            for syn in self.axn_syn_ranges_[axon as usize].clone() {
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
        self.clean();
        let presyn = presyn.dense();
        for &dend in postsyn.sparse() {
            for &syn in &self.den_synapses_[dend as usize] {
                let axon = self.syn_axons_[syn as usize];
                let presyn_active = presyn[axon as usize];
                let permanence = &mut self.syn_permanences_[syn as usize];
                let old_state = *permanence >= 0.5;
                *permanence += if presyn_active { incr } else { decr };
                let new_state = *permanence >= 0.5;
                if old_state != new_state {
                    if new_state {
                        self.den_num_connected_[dend as usize] += 1;
                    } else {
                        self.den_num_connected_[dend as usize] -= 1;
                    }
                }
            }
        }
    }

    fn clean(&mut self) {
        if self.clean_ {
            return;
        }

        self.sort_synapses();
        self.rebuild_axons();
        self.rebuild_dendrites();

        self.clean_ = true;
    }

    fn sort_synapses(&mut self) {
        // Rearrange the synapses into a new order.
        let mut syn_order = (0..self.num_syn_).into_iter();

        // Remove the dead synapses from consideration.
        let mut syn_order = syn_order.filter(|&syn| self.syn_permanences_[syn as usize] > 0.0);

        // Sort the synapses by presynaptic axon and then postsynaptic dendrite.
        let mut syn_order: Vec<_> = syn_order.collect();
        syn_order.sort_by_key(|&syn| {
            (
                self.syn_axons_[syn as usize],
                self.syn_dendrites_[syn as usize],
            )
        });

        // Filter out duplicate synapses, keep the oldest synapse, remove the rest.
        // Sort and dedup each axon's synapses by dendrite.
        syn_order.dedup_by_key(|&mut syn| {
            (
                self.syn_dendrites_[syn as usize],
                self.syn_axons_[syn as usize],
            )
        });

        // Move the synapses into their new postitions.
        self.num_syn_ = syn_order.len() as Idx;
        self.syn_axons_ = syn_order
            .iter()
            .map(|&x| self.syn_axons_[x as usize])
            .collect();
        self.syn_dendrites_ = syn_order
            .iter()
            .map(|&x| self.syn_dendrites_[x as usize])
            .collect();
        self.syn_permanences_ = syn_order
            .iter()
            .map(|&x| self.syn_permanences_[x as usize])
            .collect();
    }

    fn rebuild_axons(&mut self) {
        // Scan for contiguous ranges of synaspes from the same axon.
        self.axn_syn_ranges_ = Vec::with_capacity(self.num_axons());
        let mut start_syn: Idx = 0;
        if !self.syn_axons_.is_empty() {
            let mut cur_axon = 0;
            for syn in 0..self.num_syn_ {
                let axon = self.syn_axons_[syn as usize];
                if axon != cur_axon {
                    // Terminate this contiguous stretch.
                    self.axn_syn_ranges_.push(start_syn..syn);
                    while (self.axn_syn_ranges_.len() as Idx) < axon {
                        self.axn_syn_ranges_.push(start_syn..start_syn);
                    }
                    start_syn = syn;
                    cur_axon = axon;
                }
            }
        }
        // Terminate the final range of synapses.
        self.axn_syn_ranges_.push(start_syn..self.num_syn_);
        // Fill in any axons without synapases.
        self.axn_syn_ranges_
            .resize(self.num_axons(), self.num_syn_..self.num_syn_);
    }

    fn rebuild_dendrites(&mut self) {
        // Update the lists of synapse indices.
        for synapse_list in &mut self.den_synapses_ {
            synapse_list.clear(); // Reuse the memory allocations if possible.
        }
        self.den_synapses_.resize(self.num_dendrites(), vec![]);
        for syn in 0..self.num_synapses() {
            let dend = self.syn_dendrites_[syn] as usize;
            self.den_synapses_[dend].push(syn as Idx);
        }

        // Count the number of connected synapses.
        if self.den_num_connected_.capacity() < self.num_dendrites() {
            self.den_num_connected_ = vec![0; self.num_dendrites()];
        } else {
            self.den_num_connected_.fill(0);
            self.den_num_connected_.resize(self.num_dendrites(), 0);
        }
        for syn in 0..self.num_synapses() {
            if self.syn_permanences_[syn] >= 0.5 {
                self.den_num_connected_[self.syn_dendrites_[syn] as usize] += 1;
            }
        }
    }
}

impl std::fmt::Debug for Synapses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.clean_ {
            // todo also calc the statistics for num synapses on dendrite, and num synapses on axon.
        }
        write!(
            f,
            "Synapses (Axons: {}, Dendrites: {}, Synapses: {}, clean={})",
            self.num_axons(),
            self.num_dendrites(),
            self.num_synapses(),
            self.clean_,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut x = Synapses::new();
        assert_eq!(x.add_axons(10), 0..10);
        assert_eq!(x.add_dendrites(10), 0..10);
        let mut weights = [0.6, 0.4, 0.5].into_iter();
        x.clean(); // :(
        x.grow_selective(&mut SDR::ones(10), 3, 3, 10, || weights.next().unwrap());
        dbg!(&x);
        assert_eq!(x.num_synapses(), 3);
        let (pot, con) = x.activate(&mut SDR::ones(10));
        assert_eq!(pot, [0, 0, 0, 3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(con, [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]);

        x.add_synapse(7, 7, 0.42);
        x.add_synapse(7, 9, 0.42);
        x.add_synapse(9, 9, 0.42);
        x.add_synapse(9, 9, 0.666); // duplicate synapse
        x.activate(&mut SDR::ones(10));

        dbg!(&x);
        dbg!(&x.axn_syn_ranges_);
        dbg!(&x.syn_axons_);
        dbg!(&x.syn_dendrites_);
        dbg!(&x.syn_permanences_);
        dbg!(&x.den_synapses_);
        dbg!(&x.den_num_connected_);
        // panic!("END OF TEST");
    }

    #[test]
    fn hebbian() {
        let mut x = Synapses::new();
        x.add_axons(2);
        x.add_dendrites(1);
        x.clean(); // :(
        x.grow_selective(&mut SDR::ones(2), 0, 10, 10, || 0.5);
        let mut pattern = SDR::from_dense(vec![false, true]);
        x.hebbian(&mut pattern, &mut SDR::ones(1), 0.1, -0.1);
        let (pot, con) = x.activate(&mut SDR::ones(2));
        assert_eq!(pot, [2]);
        assert_eq!(con, [1]);
    }
}
