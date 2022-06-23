use std::iter::FromIterator;
use float_eq::float_eq;
use hashbrown::{HashMap as FastHM, HashSet};
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::thread_rng;
use crate::scpm::definition::{SCPM, SparseMatrixAttr, UNSTABLE_POLICY, XpS, XpSprPr};
use crate::*;
use crate::algorithm::motap_solver::IMOVISolver;
use crate::definition::{MDP, MDPLabel};


pub fn mdp_sparse_value_iter(
    eps: f64,
    na: usize,
    actions_start: i32,
    actions_end: i32,
    ns: usize,
    init_state: usize,
    proper_policies: &FastHM<usize, HashSet<i32>>,
    available_actions: &FastHM<usize, Vec<i32>>,
    cs_matrices: &[SparseMatrixAttr],
    rewards_map: &hashbrown::HashMap<i32, Vec<f64>>
) -> (Vec<f64>, Vec<f64>) {
    let w = vec![0.0, 0.1];
    //let mut sparse_matrices: Vec<ChannelMetaData> = Vec::new();
    let nobjs = 2;
    let mut pi: Vec<f64> = vec![-1.0; ns];
    let mut pi_new: Vec<f64> = vec![-1.0; ns];
    let mut x = vec![0f64; ns];
    let mut xnew = vec![0f64; ns]; // new value vector for agent-task pair
    let mut xtemp = vec![0f64; ns]; // a temporary vector used for copying
    let mut X: Vec<f64> = vec![0f64; ns * nobjs];
    let mut Xnew: Vec<f64> = vec![0f64; ns * nobjs];
    let mut Xtemp: Vec<f64> = vec![0f64; ns * nobjs];
    let mut epsold: Vec<f64> = vec![0f64; ns * nobjs];
    let mut inf_indices: Vec<f64>;// = Vec::new();
    let mut inf_indices_old: Vec<f64> = Vec::new();
    let mut unstable_count: i32 = 0;

    // Start off with an initial propert policy
    make_proper_policy(
        &mut pi[..],
        available_actions,
        proper_policies,
        &ns,
        &init_state
    );
    //println!("policy after randomisation: {:?}", pi);
    // construct the transition matrix generated from the random init policy
    let argmaxPinit = incremental_construct_argmax_SpPMatrix_hdd(
        &mut pi[..],
        actions_start,
        actions_end,
        ns,
        &cs_matrices[..]
    );
    // construct a rewards matrix from the random init policy
    let mut b: Vec<f64> = SCPM::incremental_construct_argmax_Rvector_hdd(
        &pi[..],
        0, // this has to be zero because we only consider one agent in this planning
        ns,
        actions_start,
        actions_end,
        &rewards_map
    );

    value_for_init_policy_sparse(&mut b[..], &mut x[..], &eps, &argmaxPinit);

    let mut epsilon: f64; // = 1.0;
    let mut policy_stable = false;
    let mut q = vec![0f64; ns * (actions_end - actions_start) as usize];
    while !policy_stable {
        policy_stable = true;
        for (ii, a) in (actions_start..actions_end).enumerate() {
            let mut vmv = vec![0f64; ns];
            sp_mv_multiply_f64(cs_matrices[ii].m, &x[..], &mut vmv);
            let mut rmv = vec![0f64; cs_matrices[ii].nr as usize];
            blas_matrix_vector_mulf64(
                &rewards_map.get(&a).unwrap()[..],
                &w[..],
                cs_matrices[ii].nr as i32,
                nobjs as i32,
                &mut rmv[..]
            );
            assert_eq!(vmv.len(), rmv.len());
            // Perform the operation R.w + P.x
            add_vecs(&rmv[..], &mut vmv[..], cs_matrices[ii].nr as i32, 1.0);
            //println!("vmv after addition with rmv:\n{:?}", vmv);
            // Add the value vector to the Q table
            update_qmat(&mut q[..], &vmv[..], ii, na as usize).unwrap();
        }
        // determine the maximum values for each state in the matrix of value estimates
        //self.max_values(&mut xnew[..], &q[..], &mut pi_new[..], ns, na as usize);
        max_values(&mut xnew[..], &q[..], &mut pi_new[..], ns, na as usize);
        copy(&xnew[..], &mut xtemp[..], ns as i32);
        // copy the new value vector to calculate epsilon
        add_vecs(&x[..], &mut xnew[..], ns as i32, -1.0);
        update_policy(&xnew[..], &eps, &mut pi[..], &pi_new[..], ns, &mut policy_stable);
        copy(&xtemp[..], &mut x[..], ns as i32);
    }
    // construct the argmax matrix to calculate the objective values
    let argmaxP = incremental_construct_argmax_SpPMatrix_hdd(
        &pi[..],
        actions_start,
        actions_end,
        ns,
        &cs_matrices[..]
    );

    let argmaxR = SCPM::incremental_construct_argmax_Rmatrix_hdd(
        &pi[..],
        ns,
        nobjs,
        actions_start,
        actions_end,
        &rewards_map
    );

    epsilon = 1.0;
    let mut epsilon_old: f64 = 1.0;
    while epsilon > eps && unstable_count < UNSTABLE_POLICY {
        for k in 0..nobjs {
            let mut vobjvec = vec![0f64; ns];
            sp_mv_multiply_f64(argmaxP.m, &X[k*ns..(k+1)*ns], &mut vobjvec[..]);
            add_vecs(&argmaxR[k*ns..(k+1)*ns], &mut vobjvec[..], ns as i32, 1.0);
            copy(&vobjvec[..], &mut Xnew[k*ns..(k+1)*ns], ns as i32);
        }
        // determine the difference between X, Xnew
        let obj_len = (ns * nobjs) as i32;
        copy(&Xnew[..], &mut Xtemp[..], obj_len);
        add_vecs(&Xnew[..], &mut X[..], obj_len, -1.0);
        epsilon = max_eps(&X[..]);
        inf_indices = X.iter()
            .zip(epsold.iter())
            .enumerate()
            .filter(|(_ix, (x, z))| float_eq!(**x - **z, 0., abs <= eps) && **x != 0.)
            .map(|(ix, _)| ix as f64)
            .collect::<Vec<f64>>();

        if inf_indices.len() == inf_indices_old.len() {
            if inf_indices.iter().zip(inf_indices_old.iter()).all(|(val1, val2)| val1 == val2) {
                //println!("eps: {} eps old: {}, inf: {:?}", epsilon, epsilon_old, inf_indices);
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
        //println!("{:?}", t5.elapsed().as_secs_f64());
        copy(&X[..], &mut epsold[..], obj_len);
        // Copy X <- Xnew
        copy(&Xtemp[..], &mut X[..], obj_len);
        // copy the unstable indices
        inf_indices_old = inf_indices;
        epsilon_old = epsilon;
    }
    if unstable_count >= UNSTABLE_POLICY {
        //println!("inf indices: {:?}", inf_indices_old);
        for ix in inf_indices_old.iter() {
            if X[*ix as usize] < 0. {
                X[*ix as usize] = -f32::MAX as f64;
            }
        }
    }
    let mut objvals = Vec::new();
    for k in 0..nobjs {
        objvals.push(X[k * ns + init_state]);
    }
    (pi, objvals)
}

fn make_proper_policy(
    pi: &mut [f64],
    available_actions: &hashbrown::HashMap<usize, Vec<i32>>,
    proper_actions: &hashbrown::HashMap<usize, HashSet<i32>>,
    ns: &usize,
    init_state: &usize
) {

    for s in 0..*ns {
        match proper_actions.get(&s) {
            Some(actions) => {
                if s != *init_state &&
                    actions.iter().any(|x| *x == -1) { // i.e this is the task handover state, like suc or fai
                    pi[s] = 0.;
                } else {
                    pi[s] = *actions
                        .into_iter()
                        .choose(&mut thread_rng())
                        .unwrap() as f64;
                }
            }
            None => {
                // randomly select any action
                let act = available_actions
                    .get(&s)
                    .unwrap()
                    .choose(&mut thread_rng())
                    .unwrap();
                pi[s] = *act as f64;
            }
        }
    }
}

fn max_eps(x: &[f64]) -> f64 {
    *x.iter().max_by(|a, b| a.partial_cmp(&b).expect("No NaNs allowed")).unwrap()
}

fn incremental_construct_argmax_SpPMatrix_hdd(
    pi: &[f64],
    actions_start: i32,
    actions_end: i32,
    m: usize,
    matrices: &[SparseMatrixAttr]
) -> SparseMatrixAttr {
    // transpose the matrices and decompose them into CSS parts
    let mut transposes: FastHM<i32, SparseMatrixComponents> = FastHM::new();

    for (ix, action) in (actions_start..actions_end).enumerate() {
        transposes.insert(
            action,
            deconstruct(transpose(matrices[ix].m, matrices[ix].nnz as i32), matrices[ix].nnz, matrices[ix].nr as usize)
        );
    }
    let mut argmax_i: Vec<i32> = Vec::new();
    let mut argmax_j: Vec<i32> = Vec::new();
    let mut argmax_vals: Vec<f64> = Vec::new();
    let actions: Vec<i32> = (actions_start..actions_end).collect();
    //
    for c in 0..m {
        let matcomp = transposes.get(&actions[pi[c] as usize]).unwrap();
        let p = &matcomp.p;
        let i = &matcomp.i;
        let x = &matcomp.x;
        if p[c + 1] - p[c] > 0 {
            for r in p[c]..p[c + 1] {
                argmax_j.push(i[r as usize]);
                argmax_i.push(c as i32);
                argmax_vals.push(x[r as usize]);
            }
        }
    }
    let nnz = argmax_vals.len();
    let argmaxT = create_sparse_matrix(
        m as i32,
        m as i32,
        &mut argmax_i[..],
        &mut argmax_j[..],
        &mut argmax_vals[..]
    );
    let argmaxA = convert_to_compressed(argmaxT);

    SparseMatrixAttr {
        m: argmaxA,
        nr: m,
        nc: m,
        nnz: nnz
    }
}

#[allow(non_snake_case)]
fn update_qmat(q: &mut [f64], v: &[f64], row: usize, nr: usize) -> Result<(), String>{
    for (ii, val) in v.iter().enumerate() {
        q[ii * nr + row] = *val;
    }
    Ok(())
}

fn max_values(x: &mut [f64], q: &[f64], pi: &mut [f64], ns: usize, na: usize) {
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

fn update_policy(eps: &[f64], thresh: &f64, pi: &mut [f64], pi_new: &[f64],
                 ns: usize, policy_stable: &mut bool) {
    for ii in 0..ns {
        if eps[ii] > *thresh {
            // println!("ix: {}, eps {}, action: {:?}", ii, eps[ii], pi_new[ii]);
            // update the action in pi with pnew
            pi[ii] = pi_new[ii];
            *policy_stable = false
        }
    }
}

fn value_for_init_policy_sparse(
    b: &mut [f64],
    x: &mut [f64],
    eps: &f64,
    argmaxP: &SparseMatrixAttr
) {
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
        epsilon = max_eps(&x[..]);
        //println!("eps: {:?}", epsilon);
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
        //println!("inf indices: {:?}", inf_indices_old);
        for ix in inf_indices_old.iter() {
            if x[*ix as usize] < 0. {
                x[*ix as usize] = -f32::MAX as f64;
            }
        }
    }
}

/// For a given MDP, for each state determine the set of available actions
pub fn set_available_actions(
    mdp: &MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>,
    actions_start: i32,
    actions_end: i32
) -> FastHM<usize, Vec<i32>> {
    let mut mdp_available_actions: FastHM<usize, Vec<i32>> = FastHM::new();

    // for each state of the mdp get the available actions for each of the states
    // get the mdp first
    for state in mdp.states.iter() {
        let state_idx = *mdp.state_mapping.get(state).unwrap();
        let mut action_set: Vec<i32> = Vec::new();
        //println!("scpm action range: [{}..{}]", self.actions.start, self.actions.end);
        for action in actions_start..actions_end {
            match mdp.transitions.get(&(*state, action)) {
                None => {}
                Some(_) => {
                    //if z.iter().map(|(_, p)|)
                    action_set.push(action);
                }
            }
        }
        mdp_available_actions.insert(state_idx, action_set);
    }
    mdp_available_actions
}

/// For a given MDP calculates the set of proper policies
pub fn proper_policies(
    mdp: &mut MDP<(i32, i32), Vec<(i32, i32)>, Vec<((i32, i32), f64)>, [f64; 2]>
) -> FastHM<usize, HashSet<i32>> {
    let mut stack: Vec<(i32, i32)> = Vec::new();
    let mut proper_policies: FastHM<usize, HashSet<i32>> = FastHM::new();
    // starting with the initial state
    //
    // get the done states of the MDP, these are the ones we know will end in rewards
    // satisfying the stochastic shortest path
    let dones = mdp.get_labels(MDPLabel::Done);
    //
    // this algorithm needs to be a little bit different, we need to start with
    // of the final states and then work backwards
    //
    let mut visited: Vec<bool> = vec![false; mdp.states.len()];
    stack.extend_from_slice(&dones[..]);
    //
    while !stack.is_empty() {
        // this is all about getting the previous state
        let obs_state = stack.pop().unwrap();
        // get the predecessor states of this MDP state and the actions
        if !visited[*mdp.state_mapping.get(&obs_state).unwrap()] {
            match mdp.pred_transitions.get(&obs_state) {
                Some(preds) => {
                    for pred in preds.iter() {
                        let pred_idx = match mdp.state_mapping.get(&pred.state) {
                            None => { panic!("state not found {:?}", &pred.state) }
                            Some(x) => { x }
                        };
                        match proper_policies.get_mut(&pred_idx) {
                            None => {
                                proper_policies
                                    .insert(
                                        *mdp.state_mapping.get(&pred.state).unwrap(),
                                        HashSet::from_iter(vec![pred.action].into_iter())
                                    );
                            }
                            Some(x) => {
                                x.insert(pred.action);
                            }
                        }
                        stack.push(pred.state);
                    }

                }
                None => {
                    assert_eq!(mdp.init_state, obs_state);
                }

            }
            visited[*mdp.state_mapping.get(&obs_state).unwrap()] = true;
        }
    }
    proper_policies
}


pub fn construct_spblas_and_rewards(
    mdp: MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>,
    act_start: i32,
    act_end: i32
) -> (HashMap<i32, COO>, HashMap<i32, Vec<f64>>) {
    let mut transition_matrices: HashMap<i32, COO> = HashMap::new();
    let mut rewards_matrics: HashMap<i32, Vec<f64>> = HashMap::new();
    let msize = mdp.states.len();
    let nsize = msize;
    for action in act_start..act_end {
        // gather the rows
        let mut rewards: Vec<f64> = vec![-f32::MAX as f64; msize * 2];
        let mut r: Vec<i32> = Vec::new();
        let mut c: Vec<i32> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for state in mdp.states.iter().filter(|(s, q)| *s!=-1 && *q!=-1) {
            // get the state mapping
            let row_idx = mdp.state_mapping.get(state).unwrap();
            match mdp.transitions.get(&(*state, action)) {
                None => { }
                Some(v) => {
                    for (sprime, p) in v.iter() {
                        let col_idx = mdp.state_mapping.get(sprime).unwrap();
                        r.push(*row_idx as i32);
                        c.push(*col_idx as i32);
                        vals.push(*p);
                    }
                }
            }
            // in the incremental matrix generation we need to construct rewards as well as
            // the MDP is being consumed by the thread pool closure
            match mdp.rewards.get(&(*state, action)) {
                None => {
                    // if the reward value is None, then this means that there is no action
                    // enabled. This essentially means that the value should be neg inf
                }
                Some(reward) => {
                    // Either there is an action but its value is -infinity or some real
                    // reward exists. If it is the case that a reward is found but that
                    // reward is -inf, then all other agent, task rewards for this matrix row
                    // should also be -inf. The above essentially means that the action is
                    // not enabled for this state
                    rewards[*row_idx] = reward[0];
                    rewards[msize + *row_idx] = reward[1];
                }
            }
        }

        // construct a new sparse matrix to save
        let nnz = vals.len();
        //self.sparse_transitions_matrix_address_values.insert(((agent, task), action),
        //                                                     SpMatrixFmt::new(r, c, vals));
        //let A = convert_to_compressed(T);
        //let S = cs_to_rust_and_destroy(A, nnz as i32, msize as i32, nsize as i32);
        // design choice: save all of the MDPs to disk
        // fmt: agent, task, action
        let S = COO {
            nzmax: nnz as i32,
            nr: msize as i32,
            nc: nsize as i32,
            i: r,
            j: c,
            x: vals,
            nz: nnz as i32
        };
        transition_matrices.insert(action, S);
        rewards_matrics.insert(action, rewards);
    }
    //}
    //self.sparse_transitions_matrix_address_values = sp_mat_addr;
    //self.sparse_transition_matrices = sparse_matrices;
    (transition_matrices, rewards_matrics)
}

pub fn mdp_rewards_fn(
    mdp: &mut MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>,
    actions_start: i32,
    actions_end: i32
) {
    let ec =  mdp.get_labels(MDPLabel::EC).to_vec();
    let accepting = mdp.get_labels(MDPLabel::Suc).to_vec();
    //println!("EC: {:?}", ec);
    //println!("acc: {:?}", accepting);
    for done in ec.iter() {
        for action in actions_start..actions_end {
            match mdp.rewards.get_mut(&(*done, action)) {
                Some(r) => {
                    if r[0] != -f32::MAX as f64 {
                        r[0] = 0f64;
                    }
                }
                None => { }
            }
        }
    }
    //}
    for acc in accepting.iter() {
        for action in actions_start..actions_end {
            match mdp.rewards.get_mut(&(*acc, action)) {
                Some(r) => { r[0] = 0f64; }
                None => { }
            }
        }
    }
}