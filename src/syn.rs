use crate::{Idx, SDR};
use rand::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::Range;

pub struct Synapses {
    minimum_incidence: f32,
    connected_incidence: f32,
    saturated_incidence: f32,
    clean: bool,
    seed: u64,

    num_synapses: usize,
    num_axons: usize,
    num_dendrites: usize,

    // The synapses are sorted by presynaptic axon and then postsynaptic dendrite.

    // Parallel arrays of synapse data.
    syn_axons: Vec<Idx>,
    syn_dendrites: Vec<Idx>,
    syn_incidence: Vec<f32>,
    syn_weights: Vec<f32>,

    // Parallel arrays of axon data.
    axn_syn_ranges: Vec<Range<Idx>>,

    // Parallel arrays of dendrite data.
    den_synapses: Vec<Vec<Idx>>,
    den_sum_weights: Vec<f32>,
}

impl Synapses {
    pub fn new(connected_incidence: f32, saturated_incidence: f32, seed: Option<u64>) -> Self {
        assert!(0.0 <= connected_incidence && connected_incidence <= 1.0);
        assert!(0.0 <= saturated_incidence && saturated_incidence <= 1.0);
        assert!(connected_incidence <= saturated_incidence);
        // let syn_weight_halfway = 0.5 * (saturated_incidence + connected_incidence);
        // let syn_weight_slope = 1.0 / (saturated_incidence - connected_incidence);
        return Self {
            minimum_incidence: connected_incidence * 0.1,
            connected_incidence: connected_incidence,
            saturated_incidence: saturated_incidence,
            clean: false,
            seed: seed.unwrap_or_else(rand::random),
            num_synapses: 0,
            num_axons: 0,
            num_dendrites: 0,
            syn_axons: vec![],
            syn_dendrites: vec![],
            syn_incidence: vec![],
            syn_weights: vec![],
            axn_syn_ranges: vec![],
            den_synapses: vec![],
            den_sum_weights: vec![],
        };
    }

    pub fn reset(&mut self) {}

    pub fn seed(&self) -> u64 {
        return self.seed;
    }

    pub fn num_axons(&self) -> usize {
        return self.num_axons;
    }

    pub fn num_dendrites(&self) -> usize {
        return self.num_dendrites;
    }

    pub fn num_synapses(&self) -> usize {
        return self.num_synapses;
    }

    pub fn add_axons(&mut self, num_axons: usize) -> Range<usize> {
        self.clean = false;
        let start_range = self.num_axons;
        self.num_axons += num_axons;
        return start_range..self.num_axons;
    }

    pub fn add_dendrites(&mut self, num_dendrites: usize) -> Range<usize> {
        self.clean = false;
        let start_range = self.num_dendrites;
        self.num_dendrites += num_dendrites;
        self.den_sum_weights.resize(self.num_dendrites, 0.0);
        return start_range..self.num_dendrites;
    }

    pub fn add_synapse(&mut self, axon: Idx, dendrite: Idx, initial_rate: f32) {
        debug_assert!(axon < self.num_axons() as Idx);
        debug_assert!(dendrite < self.num_dendrites() as Idx);
        self.clean = false;
        let syn = self.num_synapses;
        self.num_synapses += 1;
        self.syn_axons.push(axon);
        self.syn_dendrites.push(dendrite);
        self.syn_incidence.push(initial_rate);
        let weight = self.weight_function(initial_rate);
        self.syn_weights.push(weight);
    }

    pub fn weight_function(&self, incidence_rate: f32) -> f32 {
        let sigmoid_halfway = 0.5 * (self.saturated_incidence + self.connected_incidence);
        let sigmoid_slope = 1.0 / (self.saturated_incidence - self.connected_incidence);
        return 1.0 / (1.0 + (-4.0 * sigmoid_slope * (incidence_rate - sigmoid_halfway)).exp());
    }

    pub fn get_sum_weights(&self) -> &[f32] {
        debug_assert!(self.clean);
        return self.den_sum_weights.as_slice();
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
        let synapses = &self.den_synapses[dendrite as usize];
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
    pub fn grow_competitive(&mut self, axons: &mut SDR, dendrite: Idx, potential_pct: f32) {
        // Calculate the potential pool of presynapses for this dendrite.
        let estimated_pool_size = 2 * (potential_pct * axons.num_active() as f32).round() as usize;
        let mut potential_pool = Vec::with_capacity(estimated_pool_size);
        // Hash the dendrite-axon pair into a stable, unique, and psuedo-random number.
        // Use that number to decide which synapses are potentially connected.
        let mut hash_state = DefaultHasher::new();
        self.seed.hash(&mut hash_state);
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
            self.add_synapse(axon, dendrite, self.connected_incidence);
        }
    }

    pub fn activate(&mut self, activity: &mut SDR) -> (Vec<u32>, Vec<f32>) {
        debug_assert!(activity.num_cells() == self.num_axons());
        self.clean();

        let mut potential = vec![0; self.num_dendrites()];
        let mut connected = vec![0.0; self.num_dendrites()];

        for &axon in activity.sparse().iter() {
            for syn in self.axn_syn_ranges[axon as usize].clone() {
                let dend = self.syn_dendrites[syn as usize];
                potential[dend as usize] += 1;
                connected[dend as usize] += self.syn_weights[syn as usize];
            }
        }
        return (potential, connected);
    }

    pub fn learn(&mut self, axons: &mut SDR, dendrites: &mut SDR, period: f32) {
        assert!(axons.num_cells() == self.num_axons());
        assert!(dendrites.num_cells() == self.num_dendrites());
        self.clean();
        let alpha = 1.0 - (-1.0 / period).exp();
        // dbg!(alpha);
        let axons = axons.dense();
        for &dend in dendrites.sparse() {
            let mut sum_weights = 0.0;
            for &syn in &self.den_synapses[dend as usize] {
                let axon = self.syn_axons[syn as usize];
                let axon_active = axons[axon as usize];
                let rate = &mut self.syn_incidence[syn as usize];
                let target = if axon_active { 1.0 } else { 0.0 };
                *rate += alpha * (target - *rate);
                let rate = *rate;
                let weight = self.weight_function(rate);
                self.syn_weights[syn as usize] = weight;
                sum_weights += weight;
            }
            self.den_sum_weights[dend as usize] = sum_weights;
        }
    }

    pub fn clean(&mut self) {
        if self.clean {
            return;
        }

        self.sort_synapses();
        self.rebuild_axons();
        self.rebuild_dendrites();

        self.clean = true;
    }

    fn sort_synapses(&mut self) {
        // Rearrange the synapses into a new order.
        let mut syn_order = (0..self.num_synapses as Idx).into_iter();

        // Remove the dead synapses from consideration.
        let mut syn_order = syn_order.filter(|&syn| self.syn_incidence[syn as usize] >= self.minimum_incidence);

        // Sort the synapses by presynaptic axon and then postsynaptic dendrite.
        let mut syn_order: Vec<_> = syn_order.collect();
        syn_order.sort_by_key(|&syn| (self.syn_axons[syn as usize], self.syn_dendrites[syn as usize]));

        // Filter out duplicate synapses, keep the oldest synapse, remove the rest.
        // Sort and dedup each axon's synapses by dendrite.
        syn_order.dedup_by_key(|&mut syn| (self.syn_dendrites[syn as usize], self.syn_axons[syn as usize]));

        // Move the synapses into their new postitions.
        self.num_synapses = syn_order.len();
        self.syn_axons = syn_order.iter().map(|&x| self.syn_axons[x as usize]).collect();
        self.syn_dendrites = syn_order.iter().map(|&x| self.syn_dendrites[x as usize]).collect();
        self.syn_incidence = syn_order.iter().map(|&x| self.syn_incidence[x as usize]).collect();
        self.syn_weights = syn_order.iter().map(|&x| self.syn_weights[x as usize]).collect();
    }

    fn rebuild_axons(&mut self) {
        // Scan for contiguous ranges of synaspes from the same axon.
        self.axn_syn_ranges = Vec::with_capacity(self.num_axons());
        let mut start_syn: Idx = 0;
        if !self.syn_axons.is_empty() {
            let mut cur_axon = 0;
            for syn in 0..self.num_synapses() as Idx {
                let axon = self.syn_axons[syn as usize];
                if axon != cur_axon {
                    // Terminate this contiguous stretch.
                    self.axn_syn_ranges.push(start_syn..syn);
                    while (self.axn_syn_ranges.len() as Idx) < axon {
                        self.axn_syn_ranges.push(start_syn..start_syn);
                    }
                    start_syn = syn;
                    cur_axon = axon;
                }
            }
        }
        // Terminate the final range of synapses.
        self.axn_syn_ranges.push(start_syn..self.num_synapses() as Idx);
        // Fill in any axons without synapases.
        self.axn_syn_ranges
            .resize(self.num_axons(), self.num_synapses() as Idx..self.num_synapses() as Idx);
    }

    fn rebuild_dendrites(&mut self) {
        // Update the lists of synapse indices.
        for synapse_list in &mut self.den_synapses {
            synapse_list.clear(); // Reuse the memory allocations if possible.
        }
        self.den_synapses.resize(self.num_dendrites(), vec![]);
        for syn in 0..self.num_synapses() {
            let dend = self.syn_dendrites[syn] as usize;
            self.den_synapses[dend].push(syn as Idx);
        }

        // Count the number of connected synapses.
        if self.den_sum_weights.capacity() < self.num_dendrites() {
            self.den_sum_weights = vec![0.0; self.num_dendrites()];
        } else {
            self.den_sum_weights.fill(0.0);
            self.den_sum_weights.resize(self.num_dendrites(), 0.0);
        }
        for syn in 0..self.num_synapses() {
            let den = self.syn_dendrites[syn] as usize;
            self.den_sum_weights[den] += self.syn_weights[syn];
        }
    }
}

fn mean_std(data: &[f32]) -> (f32, f32) {
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    return (mean, var.sqrt());
}

impl std::fmt::Debug for Synapses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Synapses ({} Axons, {} Dendrites, {} Synapses)",
            self.num_axons(),
            self.num_dendrites(),
            self.num_synapses(),
        );
        if self.clean {
            let pot_count: Vec<_> = self.den_synapses.iter().map(|x| x.len() as f32).collect();
            let con_count: Vec<_> = self
                .den_synapses
                .iter()
                .map(|x| x.iter().map(|&s| self.syn_weights[s as usize]).sum::<f32>())
                .collect();

            let (pot_mean, pot_std) = mean_std(&pot_count);
            let (con_mean, con_std) = mean_std(&con_count);

            writeln!(f, "\nDendrite     |  min  |  max  |  mean |  std  |",)?;
            writeln!(
                f,
                "Num Synapses | {:^5} | {:^5} | {:>5.1} | {:>5.1} |",
                pot_count.iter().fold(f32::NAN, |a, b| a.min(*b)),
                pot_count.iter().fold(f32::NAN, |a, b| a.max(*b)),
                pot_mean,
                pot_std
            )?;
            writeln!(
                f,
                "Sum Weights  | {:^5.1} | {:^5.1} | {:>5.1} | {:>5.1} |",
                con_count.iter().fold(f32::NAN, |a, b| a.min(*b)),
                con_count.iter().fold(f32::NAN, |a, b| a.max(*b)),
                con_mean,
                con_std
            )?;
        }
        Ok(())
    }
}
// Display message is the same as Debug message.
impl std::fmt::Display for Synapses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self,)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let mut x = Synapses::new(0.05, 0.1, None);
        assert_eq!(x.add_axons(10), 0..10);
        assert_eq!(x.add_dendrites(10), 0..10);
        let mut weights = [10.0, 0.001, 10.0].into_iter();
        x.clean(); // :(
        x.grow_selective(&mut SDR::ones(10), 3, 3, 10, || weights.next().unwrap());
        x.clean();
        dbg!(&x);
        assert_eq!(x.num_synapses(), 2);
        let (pot, con) = x.activate(&mut SDR::ones(10));
        assert_eq!(pot, [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(con, [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        x.add_synapse(7, 7, 0.42);
        x.add_synapse(7, 9, 0.42);
        x.add_synapse(9, 9, 0.42);
        x.add_synapse(9, 9, 0.666); // duplicate synapse
        x.activate(&mut SDR::ones(10));

        dbg!(&x);
        dbg!(&x.axn_syn_ranges);
        dbg!(&x.syn_axons);
        dbg!(&x.syn_dendrites);
        dbg!(&x.syn_incidence);
        dbg!(&x.syn_weights);
        dbg!(&x.den_synapses);
        dbg!(&x.den_sum_weights);
        // panic!("END OF TEST");
    }

    #[test]
    fn learn() {
        let num_syn = 10;
        let mut x = Synapses::new(0.05, 0.1, None);
        x.add_axons(num_syn);
        x.add_dendrites(num_syn);
        for syn in 0..num_syn {
            x.add_synapse(syn as Idx, syn as Idx, f32::EPSILON);
        }

        let inp_sdr = || {
            SDR::from_dense(
                (0..num_syn)
                    .map(|x| rand::random::<f32>() < x as f32 / num_syn as f32)
                    .collect(),
            )
        };

        for train in 0..10_000 {
            x.learn(&mut inp_sdr(), &mut SDR::ones(num_syn), 1000.0);
        }
        dbg!(&x.syn_incidence);

        let atol = 0.05;
        for (idx, &x) in x.syn_incidence.iter().enumerate() {
            let exact = idx as f32 / num_syn as f32;
            assert!(exact - atol <= x && x <= exact + atol);
        }
    }
}
