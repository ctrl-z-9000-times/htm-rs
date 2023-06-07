use crate::{Encoder, Idx, PurkinjeCells, SpatialPooler, SDR};
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
        purkinje_num_cells: usize,
        purkinje_num_active: usize,
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
        let mut purkinje = Vec::with_capacity(output_spec.len());
        for (min, max, res) in &output_spec {
            purkinje.push(PurkinjeCells::new(
                num_steps,
                purkinje_num_cells,
                purkinje_num_active,
                purkinje_active_thresh,
                purkinje_potential_pct,
                purkinje_learning_period,
                purkinje_coincidence_ratio,
            ));
        }
        //
        let output_adapters: Vec<_> = output_spec
            .iter()
            .map(|(min, max, res)| Encoder::new_scalar(mossy_num_active, *min, *max, *res))
            .collect();
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
        let mut input_ref_sdr: Vec<_> = input_sdr.iter_mut().collect();
        let mut mossy_fibers = SDR::concatenate(&mut input_ref_sdr);

        // Encode the outputs.
        let mut climbing_fibers = SDR::random(99, 0.02);

        // Run the neural network.
        let mut x = self.granule.advance(&mut mossy_fibers, outputs.is_some());
        // let mut y = self.purkinje.advance(x, Some(climbing_fibers));

        // Decode the Purkinje cell activity into real numbers.
        // return predictions;
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        // let mut x = Cerebellum::new();

        // x.reset();

        // x.advance();
    }
}
