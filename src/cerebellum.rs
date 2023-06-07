use crate::{Encoder, PurkinjeCells, SpatialPooler, SDR};
use pyo3::prelude::*;

#[pyclass]
pub struct Cerebellum {
    input_adapters: Vec<Encoder>,
    output_adapters: Vec<Encoder>,
    granule: SpatialPooler,
    purkinje: Vec<PurkinjeCells>,
}

#[pymethods]
impl Cerebellum {
    #[new]
    pub fn new(
        num_steps: usize,
        input_spec: Vec<(f32, f32, f32)>,
        output_spec: Vec<(f32, f32, f32)>,
        mossy_num_active: usize,
        granule_num_cells: usize,
        granule_num_active: usize,
        granule_active_thresh: usize,
        granule_potential_pct: f32,
        granule_learning_period: f32,
        granule_coincidence_ratio: f32,
        granule_homeostatic_period: f32,
        purkinje_active_thresh: usize,
        purkinje_potential_pct: f32,
        purkinje_learning_period: f32,
        purkinje_coincidence_ratio: f32,
    ) -> Self {
        //
        let input_adapters: Vec<_> = input_spec
            .iter()
            .map(|(min, max, res)| Encoder::new_scalar(mossy_num_active, *min, *max, *res))
            .collect();
        //
        let mut granule = SpatialPooler::new(
            granule_num_cells,
            granule_num_active,
            granule_active_thresh,
            granule_potential_pct,
            granule_learning_period,
            granule_coincidence_ratio,
            granule_homeostatic_period,
        );
        //
        let output_adapters: Vec<_> = output_spec
            .iter()
            .map(|(min, max, res)| Encoder::new_scalar(mossy_num_active, *min, *max, *res))
            .collect();
        //
        let mut purkinje = Vec::with_capacity(output_spec.len());
        for enc in &output_adapters {
            purkinje.push(PurkinjeCells::new(
                num_steps,
                enc.num_cells(),
                enc.num_active(),
                purkinje_active_thresh,
                purkinje_potential_pct,
                purkinje_learning_period,
                purkinje_coincidence_ratio,
            ));
        }
        //
        return Self {
            input_adapters,
            output_adapters,
            granule,
            purkinje,
        };
    }

    pub fn num_inputs(&self) -> usize {
        return self.input_adapters.len();
    }

    pub fn num_outputs(&self) -> usize {
        return self.output_adapters.len();
    }

    pub fn reset(&mut self) {
        self.granule.reset();
        for x in &mut self.purkinje {
            x.reset()
        }
    }

    pub fn advance(&mut self, inputs: Vec<f32>, outputs: Option<Vec<f32>>) -> Vec<f32> {
        assert!(inputs.len() == self.num_inputs());

        // Encode the inputs.
        let mut input_sdr: Vec<_> = inputs
            .iter()
            .zip(&self.input_adapters)
            .map(|(&value, enc)| enc.encode(value))
            .collect();
        let mut mossy_fibers = SDR::concatenate(&mut input_sdr.iter_mut().collect::<Vec<_>>());

        // Run the granule cells.
        let mut parallel_fibers = self.granule.advance(&mut mossy_fibers, outputs.is_some());

        // Run the purkinje cells with supervised learning.
        let mut purkinje_fibers = Vec::with_capacity(self.num_outputs());
        if let Some(outputs) = &outputs {
            assert!(outputs.len() == self.num_outputs());
            for ((cells, adapter), &value) in self.purkinje.iter_mut().zip(&self.output_adapters).zip(outputs) {
                let correct = adapter.encode(value);
                purkinje_fibers.push(cells.advance(parallel_fibers.clone(), Some(correct)));
            }
        } else {
            // Run the purkinje cells with no learning.
            for cells in &mut self.purkinje {
                purkinje_fibers.push(cells.advance(parallel_fibers.clone(), None));
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let input_spec = vec![(0.0, 1.0, 0.1)];
        let output_spec = vec![(0.0, 1.0, 0.1)];
        let mut x = Cerebellum::new(
            0,           // num_steps
            input_spec,  // input_spec
            output_spec, // output_spec
            20,          // mossy_num_active
            10_000,      // granule_num_cells
            50,          // granule_num_active
            5,           // granule_active_thresh
            0.05,        // granule_potential_pct
            20.0,        // granule_learning_period
            20.0,        // granule_coincidence_ratio
            10000.0,     // granule_homeostatic_period
            5,           // purkinje_active_thresh
            0.2,         // purkinje_potential_pct
            20.0,        // purkinje_learning_period
            20.0,        // purkinje_coincidence_ratio
        );

        x.reset();

        let inp = vec![rand::random()];
        let out = vec![rand::random()];
        x.advance(inp, Some(out));
    }
}
