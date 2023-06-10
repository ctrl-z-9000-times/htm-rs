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
        granule_num_cells: usize,
        granule_num_active: usize,
        granule_active_thresh: usize,
        granule_potential_pct: f32,
        granule_learning_period: f32,
        granule_incidence_rate: f32,
        granule_homeostatic_period: f32,
        purkinje_active_thresh: usize,
        purkinje_potential_pct: f32,
        purkinje_learning_period: f32,
        purkinje_incidence_rate: f32,
    ) -> Self {
        //
        let input_adapters: Vec<_> = input_spec
            .iter()
            .map(|(min, max, res)| Encoder::new_scalar(input_num_active, *min, *max, *res))
            .collect();
        let num_mossy_fibers: usize = input_adapters.iter().map(|enc| enc.num_cells()).sum();
        let mossy_fibers = Stats::new(num_mossy_fibers, 100.0);
        //
        let mut granule_cells = SpatialPooler::new(
            granule_num_cells,
            granule_num_active,
            granule_active_thresh,
            granule_potential_pct,
            granule_learning_period,
            granule_incidence_rate,
            Some(granule_homeostatic_period),
            0,
        );
        let parallel_fibers = Stats::new(granule_cells.num_cells(), 100.0);
        //
        let output_adapters: Vec<_> = output_spec
            .iter()
            .map(|(min, max, res)| Encoder::new_scalar(input_num_active, *min, *max, *res))
            .collect();
        //
        let mut purkinje_cells = Vec::with_capacity(output_spec.len());
        let mut purkinje_fibers = Vec::with_capacity(output_spec.len());
        for enc in &output_adapters {
            purkinje_cells.push(SpatialPooler::new(
                enc.num_cells(),
                enc.num_active(),
                purkinje_active_thresh,
                purkinje_potential_pct,
                purkinje_learning_period,
                purkinje_incidence_rate,
                None,
                num_steps,
            ));
            purkinje_fibers.push(Stats::new(enc.num_cells(), 100.0));
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

    fn py_advance(&mut self, inputs: Vec<f32>, outputs: Option<Vec<f32>>) -> Vec<f32> {
        let output_slice = outputs.as_ref().map(|x| x.as_slice());
        return self.advance(&inputs, output_slice);
    }
}

impl Cerebellum {
    pub fn advance(&mut self, inputs: &[f32], outputs: Option<&[f32]>) -> Vec<f32> {
        assert!(inputs.len() == self.num_inputs());
        let learn = outputs.is_some();

        // Encode the inputs.
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
                purkinje_fibers.push(cells.advance(&mut parallel_fibers, false, None));
            }
        }
        for (sdr, stats) in purkinje_fibers.iter_mut().zip(&mut self.purkinje_fibers) {
            stats.update(sdr);
        }
        //
        let mut predictions = Vec::with_capacity(self.num_outputs());
        for (sdr, adapter) in purkinje_fibers.iter_mut().zip(&self.output_adapters) {
            let (mut value, confidence) = adapter.decode(sdr);
            if confidence < 0.1 {
                value = f32::NAN;
            }
            predictions.push(value);
        }
        return predictions;
    }
}

impl std::fmt::Display for Cerebellum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cerebellum",)?;
        writeln!(f, "MF -> GC {:?}", self.granule_cells.syn,)?;
        writeln!(f, "Parallel Fibers {}", self.parallel_fibers,)?;

        for pf in 0..self.num_outputs() {
            writeln!(f, "{}: GC -> PC {:?}", pf, self.purkinje_cells[pf].syn,)?;
            writeln!(f, "{}: Purkinje Fibers {}", pf, self.purkinje_fibers[pf],)?;
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
        let input_spec = vec![(0.0, 1.0, 0.1)];
        let output_spec = vec![(0.0, 1.0, 0.01)];
        let mut x = Cerebellum::new(
            0,           // num_steps
            input_spec,  // input_spec
            output_spec, // output_spec
            100,         // mossy_num_active
            10_000,      // granule_num_cells
            100,         // granule_num_active
            5,           // granule_active_thresh
            0.05,        // granule_potential_pct
            20.0,        // granule_learning_period
            0.05,        // granule_incidence_rate
            10000.0,     // granule_homeostatic_period
            5,           // purkinje_active_thresh
            0.5,         // purkinje_potential_pct
            20.0,        // purkinje_learning_period
            0.05,        // purkinje_incidence_rate
        );

        x.reset();

        let inp = vec![rand::random()];
        let out = vec![rand::random()];
        // let nan = x.advance(&inp, Some(&out));
        // assert!(nan[0].is_nan());
        let pred = x.advance(&inp, Some(&out));
        let pred = x.advance(&inp, Some(&out));
        let pred = x.advance(&inp, Some(&out));

        println!("{}", &x);
        dbg!(pred[0], out[0]);
        assert!(err(&pred, &out) < 0.02);
    }
}
