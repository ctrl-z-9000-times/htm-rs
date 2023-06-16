use crate::{Encoder, SpatialPooler, Stats, SDR};
use pyo3::prelude::*;

#[pyclass]
pub struct Cerebellum {
    input_adapters: Vec<Encoder>,
    output_adapters: Vec<Encoder>,
    granule_cells: SpatialPooler,
    purkinje_cells: Vec<SpatialPooler>,
    mossy_fibers: Stats,
    parallel_fibers: Stats,
    purkinje_fibers: Vec<Stats>,
}

#[pymethods]
impl Cerebellum {
    #[new]
    pub fn new(
        num_steps: usize,
        input_spec: Vec<(f32, f32, f32)>,
        output_spec: Vec<(f32, f32, f32)>,
        input_num_active: usize,
        output_num_active: usize,
        granule_num_cells: usize,
        granule_num_active: usize,
        granule_threshold: f32,
        granule_potential_pct: f32,
        granule_learning_period: usize,
        granule_num_patterns: usize,
        granule_weight_gain: f32,
        granule_boosting_period: usize,
        purkinje_threshold: f32,
        purkinje_potential_pct: f32,
        purkinje_learning_period: usize,
        purkinje_num_patterns: usize,
        purkinje_weight_gain: f32,
        seed: Option<u64>,
    ) -> Self {
        //
        let input_adapters: Vec<_> = input_spec
            .iter()
            .map(|(min, max, res)| Encoder::new_scalar(input_num_active, *min, *max, *res))
            .collect();
        let num_mossy_fibers: usize = input_adapters.iter().map(|enc| enc.num_cells()).sum();
        let mossy_fibers = Stats::new(100.0);
        //
        let mut granule_cells = SpatialPooler::new(
            granule_num_cells,
            granule_num_active,
            0,
            granule_threshold,
            granule_potential_pct,
            granule_learning_period,
            granule_num_patterns,
            granule_weight_gain,
            Some(granule_boosting_period),
            seed,
        );
        let parallel_fibers = Stats::new(100.0);
        //
        let output_adapters: Vec<_> = output_spec
            .iter()
            .map(|(min, max, res)| Encoder::new_scalar(output_num_active, *min, *max, *res))
            .collect();
        //
        let mut purkinje_cells = Vec::with_capacity(output_spec.len());
        let mut purkinje_fibers = Vec::with_capacity(output_spec.len());
        for enc in &output_adapters {
            purkinje_cells.push(SpatialPooler::new(
                enc.num_cells(),
                enc.num_active(),
                num_steps,
                purkinje_threshold,
                purkinje_potential_pct,
                purkinje_learning_period,
                purkinje_num_patterns,
                purkinje_weight_gain,
                None,
                Some(granule_cells.seed() + 123456),
            ));
            purkinje_fibers.push(Stats::new(100.0));
        }
        //
        return Self {
            input_adapters,
            output_adapters,
            granule_cells,
            purkinje_cells,
            mossy_fibers,
            parallel_fibers,
            purkinje_fibers,
        };
    }

    pub fn num_inputs(&self) -> usize {
        return self.input_adapters.len();
    }

    pub fn num_outputs(&self) -> usize {
        return self.output_adapters.len();
    }

    pub fn reset(&mut self) {
        self.granule_cells.reset();
        for x in &mut self.purkinje_cells {
            x.reset()
        }
    }

    #[pyo3(name = "advance")]
    fn py_advance(&mut self, inputs: Vec<f32>, outputs: Option<Vec<f32>>) -> Vec<f32> {
        let output_slice = outputs.as_ref().map(|x| x.as_slice());
        return self.advance(&inputs, output_slice);
    }
}

impl Cerebellum {
    pub fn advance(&mut self, inputs: &[f32], outputs: Option<&[f32]>) -> Vec<f32> {
        assert!(inputs.len() == self.num_inputs());
        let learn = outputs.is_some();

        // Encode the inputs into SDRs.
        let mut input_sdr: Vec<_> = inputs
            .iter()
            .zip(&self.input_adapters)
            .map(|(&value, enc)| enc.encode(value))
            .collect();
        let mut mossy_fibers = SDR::concatenate(&mut input_sdr.iter_mut().collect::<Vec<_>>());

        // Run the granule cells.
        let mut parallel_fibers = self.granule_cells.advance(&mut mossy_fibers, learn, None);
        if learn {
            self.parallel_fibers.update(&mut parallel_fibers);
        }

        // Run the purkinje cells with supervised learning.
        let mut purkinje_fibers = Vec::with_capacity(self.num_outputs());
        if let Some(outputs) = outputs {
            assert!(outputs.len() == self.num_outputs());
            for ((cells, adapter), &value) in self.purkinje_cells.iter_mut().zip(&self.output_adapters).zip(outputs) {
                let mut correct = adapter.encode(value);
                purkinje_fibers.push(cells.advance(&mut parallel_fibers.clone(), learn, Some(&mut correct)));
            }
        } else {
            // Run the purkinje cells with no learning.
            for cells in &mut self.purkinje_cells {
                purkinje_fibers.push(cells.advance(
                    &mut parallel_fibers,
                    false,
                    Some(&mut SDR::zeros(cells.num_outputs())),
                ));
            }
        }
        // Update the purkinje fiber statistics.
        if learn {
            for (sdr, stats) in purkinje_fibers.iter_mut().zip(&mut self.purkinje_fibers) {
                stats.update(sdr);
            }
        }

        // Decode the purkinje cell outputs real values for the user.
        let mut predictions = Vec::with_capacity(self.num_outputs());
        for (sdr, adapter) in purkinje_fibers.iter_mut().zip(&self.output_adapters) {
            let (mut value, confidence) = adapter.decode(sdr);
            predictions.push(value);
        }
        return predictions;
    }

    fn input_test_vector(&self) -> Vec<f32> {
        return self.input_adapters.iter().map(|x| x.sample()).collect();
    }
    fn output_test_vector(&self) -> Vec<f32> {
        return self.output_adapters.iter().map(|x| x.sample()).collect();
    }
}

impl std::fmt::Display for Cerebellum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cerebellum",)?;
        // TODO: Print the basic input statistics.
        writeln!(f, "Granule Cell {}", self.granule_cells,)?;
        writeln!(f, "Granule Cell Activity {}", self.parallel_fibers,)?;

        for pf in 0..self.num_outputs() {
            if self.num_outputs() > 1 {
                writeln!(f, "Output: {}", pf)?;
            }
            writeln!(f, "Purkinje Cell {}", self.purkinje_cells[pf],)?;
            writeln!(f, "Purkinje Cell Activity {}", self.purkinje_fibers[pf],)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(a: &[f32], b: &[f32]) -> f32 {
        return a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    }

    #[test]
    fn basic() {
        // I want to see how much a default-configured cerebellum can remember.
        // Let's just make a bunch of random input and output vectors.
        let input_spec = vec![(0.0, 1.0, 0.01), (0.0, 1.0, 0.01), (0.0, 1.0, 0.01)];
        let output_spec = vec![(0.0, 1.0, 0.1)];
        let mut nn = Cerebellum::new(
            0,           // num_steps
            input_spec,  // input_spec
            output_spec, // output_spec
            10,          // input_num_active
            20,          // output_num_active
            100_000,     // granule_num_cells
            50,          // granule_num_active
            0.5,         // granule_threshold
            0.25,        // granule_potential_pct
            5,           // granule_learning_period
            5,           // granule_num_patterns
            1.5,         // granule_weight_gain
            1000,        // granule_boosting_period
            0.1,         // purkinje_threshold
            0.5,         // purkinje_potential_pct
            100,         // purkinje_learning_period
            100,         // purkinje_num_patterns
            1.5,         // purkinje_weight_gain
            None,        // seed
        );

        nn.reset();

        let num_samples = if cfg!(debug_assertions) { 10 } else { 500 };
        let all_inp: Vec<_> = (0..num_samples).map(|_| nn.input_test_vector()).collect();
        let all_out: Vec<_> = (0..num_samples).map(|_| nn.output_test_vector()).collect();
        let nan = nn.advance(&all_inp[0], Some(&all_out[0]));
        assert!(nan[0].is_nan());

        // for _noise in 0..100 {
        //     nn.advance(&nn.input_test_vector(), Some(&nn.output_test_vector()));
        // }

        for _train in 0..3 {
            for (inp, out) in all_inp.iter().zip(&all_out) {
                nn.advance(inp, Some(out));
            }
        }

        nn.reset();
        println!("{}", &nn);

        for (inp, out) in all_inp.iter().zip(&all_out) {
            let pred = nn.advance(inp, None);
            println!("correct {} output {}", out[0], pred[0]);
            assert!(err(&pred, &out) < 0.1);
        }

        // panic!("END OF TEST")
    }
}
