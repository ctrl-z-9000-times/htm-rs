use crate::{Idx, SDR};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug)]
pub struct Encoder {
    num_cells: Idx,
    num_active: Idx,
    min: f32,
    max: f32,
    category: bool,
}

#[pymethods]
impl Encoder {
    #[new]
    pub fn new_scalar(num_active: usize, minimum: f32, maximum: f32, resolution: f32) -> Self {
        assert!(minimum < maximum);
        assert!(resolution > 0.0);
        let range = maximum - minimum;
        let num_bins = (range / resolution + 1.0).ceil() as usize;
        return Self {
            num_cells: (num_bins * num_active).try_into().unwrap(),
            num_active: num_active.try_into().unwrap(),
            min: minimum,
            max: maximum,
            category: false,
        };
    }

    // pub fn new_category(num_active: usize, num_categories: usize) -> Self {}

    pub fn num_cells(&self) -> usize {
        return self.num_cells as usize;
    }
    pub fn num_active(&self) -> usize {
        return self.num_active as usize;
    }
    pub fn min(&self) -> f32 {
        return self.min;
    }
    pub fn max(&self) -> f32 {
        return self.max;
    }
    pub fn category(&self) -> bool {
        return self.category;
    }

    pub fn encode(&self, value: f32) -> SDR {
        assert!(self.min <= value && value <= self.max);
        // Normalize the value into the range [0, 1).
        let value = (value - self.min) / (self.max - self.min);
        let value = value.clamp(0.0, 1.0 - f32::EPSILON);
        // Scale the value into an index in the range [0, num_cells - num_active + 1]
        let index = (value * (self.num_cells - self.num_active + 1) as f32).floor() as Idx;
        return SDR::from_sparse(
            self.num_cells as usize,
            (index..index + self.num_active as Idx).collect(),
        );
    }

    /// Returns estimated value as well as a confidence score in range [0, 1].
    pub fn decode(&self, sdr: &mut SDR) -> (f32, f32) {
        assert!(sdr.num_cells() == self.num_cells());
        let output = if self.category {
            self.decode_category(sdr)
        } else {
            self.decode_scalar(sdr)
        };
        let confidence = sdr.percent_overlap(&mut self.encode(output));
        return (output, confidence);
    }
}

impl Encoder {
    fn decode_scalar(&self, sdr: &mut SDR) -> f32 {
        // Convolve a box filter over the sparse indices.
        let box_filter = self.num_active;
        let num_active = sdr.num_active();
        let sparse = sdr.sparse();
        let mut start = 0;
        let mut end = 0;
        let mut max_cells = 0;
        let mut max_start = 0;
        while start < num_active {
            let cells_inside = sparse[end] - sparse[start] + 1;
            // Search for the span with the most active cells inside of it.
            if cells_inside > max_cells {
                max_cells = cells_inside;
                max_start = start;
            }
            // Move the end of the box forward.
            if cells_inside < box_filter && end < num_active - 1 {
                end += 1;
            }
            // Move the start of the box forward.
            else {
                start += 1;
            }
        }
        // Convert the box filter's area into a scalar.
        return sparse[max_start] as f32 / (self.num_cells - self.num_active) as f32;
    }

    fn decode_category(&self, sdr: &mut SDR) -> f32 {
        // Locate each active cell into the range [0, 1]
        let mut values = Vec::with_capacity(self.num_cells());
        for &cell in sdr.sparse() {
            values.push(cell as f32 / (self.num_cells - 1) as f32);
        }
        // If these are categories, then then round them to the nearest integer
        // and return the most common value.
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_scalar() {
        let enc = Encoder::new_scalar(10, 0.0, 1.0, 1.0);
        let mut x0 = enc.encode(0.0);
        let mut x1 = enc.encode(1.0);
        let mut x2 = enc.encode(0.5);
        let mut x3 = enc.encode(0.95);
        let mut x4 = enc.encode(0.05);
        dbg!(&enc);
        dbg!(&x0);
        dbg!(x3.sparse());
        assert!(x0.overlap(&mut x1) == 0);
        assert!(x0.overlap(&mut x2) == 5);
        assert!(x1.overlap(&mut x2) == 5);
        assert!(x1.overlap(&mut x3) == 10);
        assert!(x0.overlap(&mut x4) == 10);

        dbg!(enc.decode(&mut x0));
        dbg!(enc.decode(&mut x1));
        dbg!(enc.decode(&mut x2));
        assert!(enc.decode(&mut x0) == (0.0, 1.0));
        assert!(enc.decode(&mut x1) == (1.0, 1.0));
        assert!(enc.decode(&mut x2) == (0.5, 1.0));
    }
}
