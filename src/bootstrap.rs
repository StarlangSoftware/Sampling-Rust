use rand_new::{SeedableRng, RngExt};
use rand_new::rngs::StdRng;

pub struct Bootstrap<T: Clone> {
    pub instance_list: Vec<T>
}

impl<T: Clone> Bootstrap<T> {

    /// A constructor of Bootstrap class which takes a sample as an array of instances
    /// then creates a bootstrap sample.
    ///
    /// # Arguments
    ///
    /// * `instance_list`: Original sample
    ///
    /// returns: Bootstrap<T>
    ///
    pub fn new(instance_list: Vec<T>, seed: u64) -> Bootstrap<T> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = instance_list.len();
        let mut result : Vec<T> = Vec::with_capacity(n);
        for _ in 0..n {
            result.push(instance_list[rng.random_range(0..n)].clone());
        }
        Self{ instance_list: result}
    }

    /// getSample returns the produced bootstrap sample.
    ///
    /// returns: Vec<T, Global> Produced bootstrap sample
    ///
    pub fn get_sample(&self) -> Vec<T> {
        self.instance_list.clone()
    }

}

#[cfg(test)]
mod tests {
    use crate::bootstrap::Bootstrap;

    #[test]
    fn test_small_sample() {
        let small_sample : Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let validation = Bootstrap::new(small_sample, 42);
        assert_eq!(vec![2, 6, 3, 6, 9, 7, 10, 5, 10, 1], validation.get_sample());
    }
}