#[inline]
pub fn shape_length(shape: &[usize]) -> usize {
    shape.iter().copied()
        .chain(1..=1) // Just append 1 to the iterator
        .reduce(|a, b| a * b).unwrap_or_default()
}



#[cfg(test)]
mod tests {
    use crate::utils::Array0F;
    use super::*;

    #[test]
    fn test_len() {
        assert_eq!(shape_length(&[]), Array0F::zeros(()).len());
        assert_eq!(shape_length(&[10]), 10);
        assert_eq!(shape_length(&[5, 5]), 25);
        assert_eq!(shape_length(&[3, 3, 3]), 27);
    }
}