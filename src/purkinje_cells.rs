use crate::sp::competition;
use crate::{Idx, Synapses, SDR};
use pyo3::prelude::*;

/// This class uses the spatial pooler algorithm to learn the given sequence of
/// (input -> output) pairs. If num_steps > 0 then this predicts a future
/// output instead of the current one.
#[pyclass]
pub struct PurkinjeCells {
    num_steps: usize,
    num_cells: usize,
    num_active: usize,
    active_thresh: usize,
    potential_pct: f32,
    learning_period: f32,
    coincidence_ratio: f32,

    step: usize,
    input_buffer: Vec<SDR>,
    syn: Synapses,
}

#[pymethods]
impl PurkinjeCells {
    #[new]
    pub fn new(
        num_steps: usize,
        num_cells: usize,
        num_active: usize,
        active_thresh: usize,
        potential_pct: f32,
        learning_period: f32,
        coincidence_ratio: f32,
    ) -> Self {
        let mut syn = Synapses::new();
        syn.add_dendrites(num_cells);
        return Self {
            num_steps,
            num_cells,
            num_active,
            active_thresh,
            potential_pct,
            learning_period,
            coincidence_ratio,
            step: 0,
            input_buffer: vec![SDR::zeros(0); num_steps],
            syn,
        };
    }

    pub fn num_cells(&self) -> usize {
        return self.num_cells;
    }
    pub fn num_steps(&self) -> usize {
        return self.num_steps;
    }

    pub fn reset(&mut self) {
        self.input_buffer = vec![SDR::zeros(0); self.num_steps()];
    }

    pub fn advance(&mut self, mut input: SDR, mut output: Option<SDR>) -> SDR {
        if self.syn.num_axons() == 0 {
            self.syn.add_axons(input.num_cells());
        }
        let (connected, potential) = self.syn.activate(&mut input);

        // Process the active synapse inputs into a general "activity"
        let mut activity: Vec<_> = connected
            .iter()
            .zip(self.syn.get_num_connected())
            .map(|(&x, &nsyn)| {
                if nsyn == 0 {
                    0.0
                } else {
                    (x as f32) / (nsyn as f32) // Normalize the activity by the number of connected synapses.
                }
            })
            .collect();

        // Run the Winner-Takes-All Competition.
        let mut active = competition(&activity, self.num_active);

        // Apply the activation threshold.
        active.retain(|&cell| connected[cell as usize] > self.active_thresh as Idx);

        let mut active = SDR::from_sparse(self.num_cells, active);

        if let Some(mut output) = output {
            // Learn the association: input[t-num_steps] -> output[t]
            if self.num_steps() > 0 {
                std::mem::swap(&mut input, &mut self.input_buffer[self.step]);
                self.step = (self.step + 1) % self.num_steps(); // Rotate our index into the circular buffer.
                if input.num_cells() == 0 {
                    input = SDR::zeros(self.syn.num_axons());
                }
            }
            // Apply Hebbian learning to the correct outputs.
            let incr = 1.0 / self.learning_period;
            let decr = -incr / self.coincidence_ratio;
            self.syn.hebbian(&mut input, &mut output, incr, decr);
            // Grow new synapses.
            let w = || 0.5;
            let w = rand::random::<f32>;
            for &dend in output.sparse() {
                self.syn
                    .grow_competitive(&mut input, dend, self.potential_pct, w);
            }

            // Depress the synapses leading to the incorrect outputs.
            // let incorrect = active - output;
            // self.syn.hebbian(&mut input, &mut incorrect, decr, 0.0);
        }

        return active;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // Make an artificial sequence of SDRs to demonstrate the predictor.
        let delay = 3;
        let num_cells = 2000;
        let num_active = 40;
        let input_sdr = || SDR::random(100_000, 0.001);
        let output_sdr = || SDR::random(num_cells, num_active as f32 / num_cells as f32);
        let mut nn = PurkinjeCells::new(
            delay,      // num_steps
            num_cells,  // num_cells
            num_active, // num_active
            10,         // active_thresh
            0.3,        // potential_pct
            10.0,       // learning_period
            10.0,       // coincidence_ratio
        );

        let seq_len = if cfg!(debug_assertions) { 10 } else { 1000 };
        let mut input_seq: Vec<SDR> = (0..seq_len).map(|_| input_sdr()).collect();
        let mut output_seq: Vec<SDR> = (0..seq_len).map(|_| output_sdr()).collect();

        // Train.
        for trial in 0..10 {
            for t in 0..seq_len {
                nn.advance(input_seq[t].clone(), Some(output_seq[t].clone()));
            }
        }

        // This reset prevents the last elements of the training sequence from
        // learning the random noise.
        nn.reset();

        for noise in 0..3 * seq_len {
            nn.advance(input_sdr(), Some(output_sdr()));
        }

        // Test.
        dbg!(&nn.syn);

        for t in 0..seq_len {
            let mut prediction = nn.advance(input_seq[t].clone(), None);
            let correct = &mut output_seq[(t + delay) % seq_len];
            let mut overlap = prediction.percent_overlap(correct);
            dbg!(overlap);
            assert!(overlap >= 0.90);
        }
    }
}
