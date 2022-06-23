use crate::algorithm::motap_solver::{SolverUtils};
use crate::scpm::definition::{SCPM, SparseMatrixAttr, UNSTABLE_POLICY};
use crate::utils::number_fmts::integer_decode;
use crate::{add_vecs, copy, sp_mv_multiply_f64};
use float_eq::float_eq;

#[derive(Hash, Eq, PartialEq)]
pub struct Mantissa((u64, i16, i8));

impl Mantissa {
    pub fn new(val: f64) -> Mantissa {
        Mantissa(integer_decode(val))
    }
}

impl SolverUtils for SCPM {
    #[allow(non_snake_case)]
    fn update_qmat(&self, q: &mut [f64], v: &[f64], row: usize, nr: usize) -> Result<(), String>{
        for (ii, val) in v.iter().enumerate() {
            q[ii * nr + row] = *val;
        }
        Ok(())
    }

    fn max_values(&self, x: &mut [f64], q: &[f64], pi: &mut [f64], ns: usize, na: usize) {
        for ii in 0..ns {
            let (imax, max) = q[ii*na..(ii + 1)*na]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)|
                    a.partial_cmp(b).expect("no NaNs allowed!"))
                .unwrap();
            pi[ii] = imax as f64;
            x[ii] = *max;
        }
    }

    fn update_policy(&self, eps: &[f64], thresh: &f64, pi: &mut [f64], pi_new: &[f64],
                     ns: usize, policy_stable: &mut bool) {
        for ii in 0..ns {
            if eps[ii] > *thresh {
                pi[ii] = pi_new[ii];
                *policy_stable = false
            }
        }
    }

    fn max_eps(&self, x: &[f64]) -> f64 {
        *x.iter().max_by(|a, b| a.partial_cmp(&b).expect("No NaNs allowed")).unwrap()
    }

    #[allow(non_snake_case)]
    fn gather_init_costs(&self, X: &[f64]) -> Vec<f64> {
        let mut r: Vec<f64> = Vec::with_capacity(self.num_tasks + self.num_agents);
        for k in 0..self.num_agents + self.num_tasks {
            r.push(X[k]);
        }
        r
    }

    fn value_for_policy_sparse(
        &self,
        eps: f64,
        nobjs: usize,
        ns: &usize,
        nsprime: &usize,
        argmaxP: &SparseMatrixAttr,
        argmaxR: &[f64],
        agent: i32,
        task: i32
    ) -> (f64, f64) {

        let mut X: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mut Xnew: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mut Xtemp: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mut epsilon = 1.0;
        let mut epsilon_old: f64 = 1.0;
        let mut epsold: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mdp_init = *self.get_init_state(agent, task);
        let mut inf_indices: Vec<f64>;// = Vec::new();
        let mut inf_indices_old: Vec<f64> = Vec::new();
        let mut unstable_count: i32 = 0;

        while epsilon > eps && unstable_count < UNSTABLE_POLICY {
            for k in 0..nobjs {
                let mut vobjvec = vec![0f64; *ns];
                sp_mv_multiply_f64(argmaxP.m, &X[k*nsprime..(k+1)*nsprime], &mut vobjvec[..]);
                add_vecs(&argmaxR[k*ns..(k+1)*ns], &mut vobjvec[..], *ns as i32, 1.0);
                copy(&vobjvec[..], &mut Xnew[k*nsprime..(k+1)*nsprime], *ns as i32);
            }
            // determine the difference between X, Xnew
            let obj_len = (*nsprime * (self.num_agents + self.num_tasks)) as i32;
            copy(&Xnew[..], &mut Xtemp[..], obj_len);
            add_vecs(&Xnew[..], &mut X[..], obj_len, -1.0);
            epsilon = self.max_eps(&X[..]);
            inf_indices = X.iter()
                .zip(epsold.iter())
                .enumerate()
                .filter(|(_ix, (x, z))| float_eq!(**x - **z, 0., abs <= eps) && **x != 0.)
                .map(|(ix, _)| ix as f64)
                .collect::<Vec<f64>>();

            if inf_indices.len() == inf_indices_old.len() {
                if inf_indices.iter().zip(inf_indices_old.iter()).all(|(val1, val2)| val1 == val2) {
                    if epsilon < epsilon_old {
                        // the value function is still contracting an this is converging, therefore
                        // not unstable
                        unstable_count = 0;
                    } else {
                        unstable_count += 1;
                    }
                } else {
                    unstable_count = 0;
                }
            } else {
                unstable_count = 0;
            }
            copy(&X[..], &mut epsold[..], obj_len);
            // Copy X <- Xnew
            copy(&Xtemp[..], &mut X[..], obj_len);
            // copy the unstable indices
            inf_indices_old = inf_indices;
            epsilon_old = epsilon;
        }
        if unstable_count >= UNSTABLE_POLICY {
            for ix in inf_indices_old.iter() {
                if X[*ix as usize] < 0. {
                    X[*ix as usize] = -f32::MAX as f64;
                }
            }
        }

        (
            X[agent as usize * nsprime + mdp_init],
            X[(self.num_agents + task as usize) * nsprime + mdp_init]
        )
    }

    // b is an already weighted rewards vector i.e. R.w -> [1xns]
    fn value_for_init_policy__sparse(&self, b: &mut [f64], x: &mut [f64], eps: &f64, argmaxP: &SparseMatrixAttr) {
        let mut epsilon: f64 = 1.0;
        let mut xnew: Vec<f64> = vec![0f64; argmaxP.nc];
        let mut epsold: Vec<f64> = vec![0f64; argmaxP.nc];
        let mut unstable_count: i32 = 0;
        let mut inf_indices: Vec<f64>; // = Vec::new();
        let mut inf_indices_old: Vec<f64> = Vec::new();
        let mut epsilon_old: f64 = 1.0;
        while (epsilon > *eps) && (unstable_count < UNSTABLE_POLICY) {
            let mut vmv = vec![0f64; argmaxP.nr];
            sp_mv_multiply_f64(argmaxP.m, &x[..], &mut vmv[..]);
            add_vecs(&mut b[..], &mut vmv[..], argmaxP.nr as i32, 1.0);
            // vmv -> Xnew: but remember that dim xnew does not necessarily equal vmv
            copy(&vmv[..], &mut xnew[..argmaxP.nr], argmaxP.nr as i32);
            add_vecs(&xnew[..], &mut x[..], argmaxP.nc as i32, -1.0);
            epsilon = self.max_eps(&x[..]);
            inf_indices = x[..argmaxP.nr].iter()
                .zip(epsold.iter())
                .enumerate()
                .filter(|(_, (x, z))| float_eq!(**x - **z, 0., abs <= eps) && **x != 0.)
                .map(|(ix, _)| ix as f64)
                .collect::<Vec<f64>>();

            if inf_indices.len() == inf_indices_old.len() {
                if inf_indices.iter().zip(inf_indices_old.iter()).all(|(val1, val2)| val1 == val2) {
                    if epsilon < epsilon_old {
                        unstable_count = 0;
                    } else {
                        unstable_count += 1;
                    }
                } else {
                    unstable_count = 0;
                }
            } else {
                unstable_count = 0;
            }

            copy(&x[..], &mut epsold[..], argmaxP.nc as i32);
            // replace all of the values where x and epsold are equal with NEG_INFINITY or INFINITY
            // depending on sign
            copy(&vmv[..], &mut x[..argmaxP.nr], argmaxP.nr as i32);

            inf_indices_old = inf_indices;
            epsilon_old = epsilon;
        }
        if unstable_count >= UNSTABLE_POLICY {
            for ix in inf_indices_old.iter() {
                if x[*ix as usize] < 0. {
                    x[*ix as usize] = -f32::MAX as f64;
                }
            }
        }
    }
}