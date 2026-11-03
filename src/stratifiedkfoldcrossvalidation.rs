use rand_old::prelude::StdRng;
use rand_old::SeedableRng;
use rand_old::seq::SliceRandom; // Brings the shuffle method into scope

pub struct StratifiedKFoldCrossvalidation<T: Clone> {
    pub K: usize,
    pub instance_lists: Vec<Vec<T>>,
    pub N: Vec<usize>
}

impl<T: Clone> StratifiedKFoldCrossvalidation<T> {

    /// A constructor of StratifiedKFoldCrossvalidation class which takes a sample as an array of array of instances, a
    /// K (K in K-fold cross-validation),
    /// then shuffles the original sample.
    ///
    /// # Arguments
    ///
    /// * `instance_lists`: s Original class samples. Each element of this array is a sample only from one class.
    /// * `k`: K in K-fold cross-validation
    ///
    /// returns: StratifiedKFoldCrossvalidation<T>
    ///
    pub fn new(mut instance_lists: Vec<Vec<T>>, k: usize, seed: u64) -> StratifiedKFoldCrossvalidation<T> {
        let mut n = Vec::with_capacity(instance_lists.len());
        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..instance_lists.len() {
            instance_lists[i].shuffle(&mut rng);
            n.push(instance_lists[i].len());
        }
        Self{ K: k, instance_lists, N: n }
    }

    /// getTrainFold returns the k'th train fold in stratified K-fold cross-validation.
    ///
    /// # Arguments
    ///
    /// * `k`: index for the k'th train fold of the stratified K-fold cross-validation
    ///
    /// returns: Vec<T, Global> Produced training sample
    ///
    pub fn get_train_fold(&self, k: usize) -> Vec<T> {
        let mut train_fold = vec![];
        for i in 0..self.N.len(){
            for j in 0..(k * self.N[i] / self.K) {
                train_fold.push(self.instance_lists[i][j].clone());
            }
            for j in (k + 1) * self.N[i] / self.K..self.N[i] {
                train_fold.push(self.instance_lists[i][j].clone());
            }
        }
        train_fold
    }

    /// getTestFold returns the k'th test fold in stratified K-fold cross-validation.
    ///
    /// # Arguments
    ///
    /// * `k`: index for the k'th test fold of the stratified K-fold cross-validation
    ///
    /// returns: Vec<T, Global> Produced testing sample
    ///
    pub fn get_test_fold(&self, k: usize) -> Vec<T> {
        let mut test_fold = vec![];
        for i in 0..self.N.len() {
            for j in k * self.N[i] / self.K..(k + 1) * self.N[i] / self.K {
                test_fold.push(self.instance_lists[i][j].clone());
            }
        }
        test_fold
    }

}