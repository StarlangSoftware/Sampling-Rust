use rand_old::seq::SliceRandom; // Brings the shuffle method into scope
use rand_old::prelude::StdRng;
use rand_old::SeedableRng;

pub struct KFoldCrossvalidation<T: Clone> {
    pub K: usize,
    pub instance_list: Vec<T>,
    pub N: usize
}

impl<T: Clone> KFoldCrossvalidation<T> {

    /// A constructor of KFoldCrossValidation class which takes a sample as an array of instances, a
    /// K (K in K-fold cross-validation),
    /// then shuffles the original sample.
    ///
    /// # Arguments
    ///
    /// * `instance_list`: Original sample
    /// * `k`: K in K-fold cross-validation
    ///
    /// returns: KFoldCrossvalidation<T>
    ///
    pub fn new(mut instance_list: Vec<T>, k: usize, seed: u64) -> KFoldCrossvalidation<T> {
        let n = instance_list.len();
        let mut rng = StdRng::seed_from_u64(seed);
        instance_list.shuffle(&mut rng);
        Self{ K: k, instance_list, N: n }
    }

    /// getTrainFold returns the k'th train fold in K-fold cross-validation.
    ///
    /// # Arguments
    ///
    /// * `k`: index for the k'th train fold of the K-fold cross-validation
    ///
    /// returns: Vec<T, Global> Produced training sample
    ///
    pub fn get_train_fold(&self, k: usize) -> Vec<T> {
        let mut train_fold = vec![];
        for i in 0..(k * self.N / self.K) {
            train_fold.push(self.instance_list[i].clone());
        }
        for i in (k + 1) * self.N / self.K..self.N {
            train_fold.push(self.instance_list[i].clone());
        }
        train_fold
    }

    /// getTestFold returns the k'th test fold in K-fold cross-validation.
    ///
    /// # Arguments
    ///
    /// * `k`: index for the k'th test fold of the K-fold cross-validation
    ///
    /// returns: Vec<T, Global> Produced testing sample
    ///
    pub fn get_test_fold(&self, k: usize) -> Vec<T> {
        let mut test_fold = vec![];
        for i in k * self.N / self.K..(k + 1) * self.N / self.K {
            test_fold.push(self.instance_list[i].clone());
        }
        test_fold
    }

}

#[cfg(test)]
mod tests {
    use crate::kfoldcrossvalidation::KFoldCrossvalidation;

    #[test]
    fn test_small_sample_10_fold() {
        let small_sample : Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let validation = KFoldCrossvalidation::new(small_sample, 10, 42);
        assert_eq!(vec![8], validation.get_test_fold(0));
    }

    #[test]
    fn test_small_sample_5_fold() {
        let small_sample : Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let validation = KFoldCrossvalidation::new(small_sample, 5, 42);
        assert_eq!(vec![8, 4], validation.get_test_fold(0));
    }

    #[test]
    fn test_small_sample_2_fold() {
        let small_sample : Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let validation = KFoldCrossvalidation::new(small_sample, 2, 42);
        assert_eq!(vec![8, 4, 10, 6, 1], validation.get_test_fold(0));
    }

}