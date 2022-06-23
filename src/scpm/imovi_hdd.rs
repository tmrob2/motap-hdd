use std::collections::VecDeque;
use std::iter::FromIterator;
use std::sync::{Arc, Mutex};
use std::thread;
use hashbrown::{HashMap as FastHM, HashSet};
use std::sync::mpsc::{Receiver, sync_channel, SyncSender};
use std::time::Instant;
use float_eq::float_eq;
use indicatif::{ProgressBar, ProgressStyle};
use crate::algorithm::motap_solver::{IMOVISolver, SolverUtils};
use crate::*;
use crate::scpm::definition::{SCPM, ValueCache, SparseMatrixAttr, UNSTABLE_POLICY, SwitchTypeCache};
use crate::definition::{MDP, MDPLabel};

impl IMOVISolver for SCPM {

    fn calculate_extreme_points(
        &self,
        rx: Receiver<ChannelMetaData>,
        num_threads: usize,
        eps: f64,
        na: usize,
        w: &[f64],
        num_extreme_points: usize
    ) -> (Vec<FastHM<(i32, i32), Vec<f64>>>, FastHM<usize, Vec<f64>>) {

        // first we need to receive the appropriate data from the channel
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let nobjs = self.num_tasks + self.num_agents;
        let mut EPPi: Vec<FastHM<(i32, i32), Vec<f64>>> = vec![FastHM::new(); num_extreme_points];
        let mut switch_cache: Vec<FastHM<SwitchTypeCache, ValueCache>> = vec![FastHM::from_iter(vec![
            (SwitchTypeCache::NextTask, ValueCache::new()),
            (SwitchTypeCache::NextAgent, ValueCache::new())
        ].into_iter()); nobjs];
        let cache = Arc::new(Mutex::new(&mut switch_cache));
        let Pi = Arc::new(Mutex::new(&mut EPPi));
        let bar = ProgressBar::new((self.num_agents * self.num_tasks) as u64);
        bar.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .progress_chars("##-"));
        //let warc = Arc::new(&w[..]);
        for j in (0..self.num_tasks).rev() {
            for i in (0..self.num_agents).rev() {
                bar.inc(1);
                // Define a memory block
                {
                    let mut rewards_map: FastHM<i32, Vec<f64>> = FastHM::new();
                    let mut coo_map: FastHM<i32, COO> = FastHM::new();

                    for action in self.actions.start..self.actions.end {
                        // for each of the action receive the Sparse matrix that is required
                        // for processing
                        match rx.recv() {
                            Ok(mat) => {
                                rewards_map.insert(action, mat.R);
                                coo_map.insert(action, mat.A);
                                //sparse_matrices.push(mat);
                                //cs_matrices.push(SparseMatrixAttr {
                                //    m: sparse_to_cs(&mut mat.A),
                                //    nr: mat.A.nr as usize,
                                //    nc: mat.A.nc as usize,
                                //    nnz: mat.A.nz as usize
                                //});
                            }
                            Err(_) => {}
                        }
                    }
                    let rewards_arc = Arc::new(rewards_map);
                    let coo_arc = Arc::new(coo_map);
                    //let switch_cache_clone = Arc::new(switch_cache);
                    //let (tx2, rx2) = std::sync::mpsc::channel();
                    pool.scope(|s| {
                        let cache= &cache;
                        for e in 0..num_extreme_points {
                            //println!("a: {}, t: {}", i, j);
                            //
                            let wthread = &w;
                            //let c_e = cache[e];
                            let coo_arc_e = coo_arc.clone();
                            let rewards_arc_e = rewards_arc.clone();
                            let cache_e = cache.clone();
                            let pi_e = Pi.clone();
                            //let warc_clone = warc.clone();
                            s.spawn(move |_| {
                                //let e = &e;
                                let ep = &wthread[e * nobjs..(e + 1) * nobjs];
                                let coo_arc = coo_arc_e;
                                let rewards_arc = rewards_arc_e;
                                self.subset_value_iteration(
                                    e,
                                    ep,
                                    na,
                                    pi_e,
                                    cache_e,
                                    coo_arc,
                                    rewards_arc,
                                    &eps,
                                    i as i32,
                                    j as i32
                                );
                            });
                        }
                    });
                }
            }
        }
        let mut values_output: FastHM<usize, Vec<f64>> = FastHM::new();
        for e in 0..nobjs {
            let cached = &switch_cache[e].get(&SwitchTypeCache::NextTask).unwrap();
            let r = self.gather_init_costs(
                &cached.obj_values.as_ref().unwrap()[..]
            );
            //let wprint = &w[e * nobjs..(e + 1) * nobjs];
            //println!("w: {:?}, r: {:.2?}", wprint, r);
            values_output.insert(e, r);

        }
        (EPPi, values_output)
    }

    fn sparse_value_iter_hdd(&self, rx: Receiver<ChannelMetaData>, eps: f64, na: usize, w: &[f64])
        -> (FastHM<(i32, i32), Vec<f64>>, Vec<f64>) {
        let mut Pi: FastHM<(i32, i32), Vec<f64>> = FastHM::new();
        let mut switch_cache: FastHM<SwitchTypeCache, ValueCache> = FastHM::from_iter(vec![
            (SwitchTypeCache::NextTask, ValueCache::new()),
            (SwitchTypeCache::NextAgent, ValueCache::new())
        ].into_iter());
        assert!(na > 0);
        let bar = ProgressBar::new((self.num_agents * self.num_tasks) as u64);
        bar.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .progress_chars("##-"));
        for j in (0..self.num_tasks as i32).rev() {
            for i in (0..self.num_agents as i32).rev() {
                bar.inc(1);
                // define a memory block
                {
                    //let mut sparse_matrices: Vec<ChannelMetaData> = Vec::new();
                    let mut cs_matrices: Vec<_> = Vec::new();
                    let mut rewards_map: FastHM<i32, Vec<f64>> = FastHM::new();
                    //let mut coo_map: FastHM<i32, COO> = FastHM::new();
                    for action in self.actions.start..self.actions.end {
                        // for each of the action receive the Sparse matrix that is required
                        // for processing
                        match rx.recv() {
                            Ok(mut mat) => {
                                rewards_map.insert(action, mat.R);
                                //coo_map.insert(action, mat.A);
                                //sparse_matrices.push(mat);
                                cs_matrices.push(SparseMatrixAttr {
                                    m: sparse_to_cs(&mut mat.A),
                                    nr: mat.A.nr as usize,
                                    nc: mat.A.nc as usize,
                                    nnz: mat.A.nz as usize
                                });
                            }
                            Err(_) => {}
                        }
                    }
                    //let rewards_arc = Arc::new(rewards_map);
                    //let coo_arc = Arc::new(coo_map);
                    // construct a threadpool? collection of threads? to work on a scope involving the Arc ref

                    let ns = *self.state_dims.get(&(i, j)).unwrap(); // number of unmodified states
                    let nsprime = *self.value_vec_dims.get(&(i, j)).unwrap(); // number of total states
                    //println!("Agent: {}, Task: {}, |S|: {}, |S'|: {}", i, j, ns, nsprime);
                    // including switch transition states (links the product matrices in sequence)
                    // x, x:
                    let nobjs = self.num_agents + self.num_tasks;
                    let mut pi: Vec<f64> = vec![-1.0; ns];
                    let mut pi_new: Vec<f64> = vec![-1.0; ns];
                    let mut x = vec![0f64; nsprime];
                    let mut xnew = vec![0f64; nsprime]; // new value vector for agent-task pair
                    let mut xtemp = vec![0f64; nsprime]; // a temporary vector used for copying
                    let mut X: Vec<f64> = vec![0f64; nsprime * nobjs];
                    let mut Xnew: Vec<f64> = vec![0f64; nsprime * nobjs];
                    let mut Xtemp: Vec<f64> = vec![0f64; nsprime * nobjs];
                    let mut epsold: Vec<f64> = vec![0f64; nsprime * nobjs];
                    let mut inf_indices: Vec<f64>;// = Vec::new();
                    let mut inf_indices_old: Vec<f64> = Vec::new();
                    let mut unstable_count: i32 = 0;
                    self.update_switch_values(i, j, &mut x[..], &mut X[..],
                                              &mut Xnew[..], &switch_cache, nsprime);

                    // make a random policy
                    self.incremental_get_rand_proper_policy(
                        i, j, &mut pi[..]
                    );
                    //println!("policy after randomisation: {:?}", pi);
                    // construct the transition matrix generated from the random init policy
                    let argmaxPinit = self.incremental_construct_argmax_SpPMatrix_hdd(
                        &mut pi[..],
                        i,
                        j,
                        &cs_matrices[..]
                    );
                    // construct a rewards matrix from the random init policy
                    let mut b: Vec<f64> = Self::incremental_construct_argmax_Rvector_hdd(
                        &pi[..],
                        i,
                        ns,
                        self.actions.start,
                        self.actions.end,
                        &rewards_map
                    );
                    self.value_for_init_policy__sparse(
                        &mut b[..],
                        &mut x[..],
                        &eps,
                        &argmaxPinit
                    );

                    let mut epsilon: f64; // = 1.0;
                    let mut policy_stable = false;
                    let mut q = vec![0f64; ns * (self.actions.end - self.actions.start) as usize];
                    while !policy_stable {
                        policy_stable = true;
                        for (ii, a) in (self.actions.start..self.actions.end).enumerate() {
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
                            self.update_qmat(&mut q[..], &vmv[..], ii, na as usize).unwrap();
                        }
                        // determine the maximum values for each state in the matrix of value estimates
                        //self.max_values(&mut xnew[..], &q[..], &mut pi_new[..], ns, na as usize);
                        self.max_values(&mut xnew[..], &q[..], &mut pi_new[..], ns, na as usize);
                        copy(&xnew[..], &mut xtemp[..], nsprime as i32);
                        // copy the new value vector to calculate epsilon
                        add_vecs(&x[..], &mut xnew[..], ns as i32, -1.0);
                        self.update_policy(&xnew[..], &eps, &mut pi[..], &pi_new[..], ns, &mut policy_stable);
                        copy(&xtemp[..], &mut x[..], ns as i32);
                    }
                    let mdp_init = *self.get_init_state(i, j);
                    if i == 0 {
                        // Cache storing all quantities for the next task
                        switch_cache.get_mut(&SwitchTypeCache::NextTask).unwrap().set_value_vec(x[mdp_init]);
                    } else {
                        // Cache storing all quantities for next agent
                        switch_cache.get_mut(&SwitchTypeCache::NextAgent).unwrap().set_value_vec(x[mdp_init]);
                    }
                    // construct the argmax matrix to calculate the objective values
                    let argmaxP = self.incremental_construct_argmax_SpPMatrix_hdd(
                        &pi[..],
                        i,
                        j,
                        &cs_matrices[..]
                    );

                    let argmaxR = Self::incremental_construct_argmax_Rmatrix_hdd(
                        &pi[..],
                        ns,
                        nobjs,
                        self.actions.start,
                        self.actions.end,
                        &rewards_map
                    );

                    epsilon = 1.0;
                    let mut epsilon_old: f64 = 1.0;
                    let t2 = Instant::now();
                    while epsilon > eps && unstable_count < UNSTABLE_POLICY {
                        for k in 0..nobjs {
                            let mut vobjvec = vec![0f64; ns];
                            sp_mv_multiply_f64(argmaxP.m, &X[k*nsprime..(k+1)*nsprime], &mut vobjvec[..]);
                            add_vecs(&argmaxR[k*ns..(k+1)*ns], &mut vobjvec[..], ns as i32, 1.0);
                            copy(&vobjvec[..], &mut Xnew[k*nsprime..(k+1)*nsprime], ns as i32);
                        }
                        // determine the difference between X, Xnew
                        let obj_len = (nsprime * (self.num_agents + self.num_tasks)) as i32;
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
                    for k in 0..self.num_agents + self.num_tasks {
                        objvals.push(X[k * nsprime + mdp_init]);
                    }
                    if i == 0 {
                        switch_cache.get_mut(&SwitchTypeCache::NextTask).unwrap().set_objective_vec(&objvals[..], &(self.num_agents + self.num_tasks));
                    } else {
                        switch_cache.get_mut(&SwitchTypeCache::NextAgent).unwrap().set_objective_vec(&objvals[..], &(self.num_agents + self.num_tasks));
                    }
                    //println!("switch cache: {:?}", switch_cache);
                    Pi.insert((i, j), pi);
                    //println!("Time taken for second loop: {:?}", t4.elapsed().as_secs_f64());
                    let msg = format!("Second loop performance {:.3?}s, agent: {}, task: {}", t2.elapsed().as_secs_f64(), i, j);
                    bar.set_message(msg);
                }
            }
        }

        let cached_task = &switch_cache.get(&SwitchTypeCache::NextTask).unwrap();
        //let cached_agent = &switch_cache.get(&SwitchTypeCache::NextAgent).unwrap();
        //println!("next task cached obj values: {:?}", cached_task.obj_values);
        //println!("next agent cached obj values: {:?}", cached_agent.obj_values);
        let r = self.gather_init_costs(
            &cached_task.obj_values.as_ref().unwrap()[..]
        );
        bar.finish();
        //println!("X: {:?}", &cached.objective_vector[..]);
        (Pi, r)
    }

    fn incremental_construct_argmax_SpPMatrix_hdd(
        &self,
        pi: &[f64],
        a: i32,
        t: i32,
        matrices: &[SparseMatrixAttr]
    ) -> SparseMatrixAttr {
        // transpose the matrices and decompose them into CSS parts
        let mut transposes: FastHM<i32, SparseMatrixComponents> = FastHM::new();

        for (ix, action) in (self.actions.start..self.actions.end).enumerate() {
            transposes.insert(
                action,
                deconstruct(transpose(matrices[ix].m, matrices[ix].nnz as i32), matrices[ix].nnz, matrices[ix].nr as usize)
            );
        }
        let m = self.get_state_dims(a, t);
        let n = *self.value_vec_dims.get(&(a, t)).unwrap();
        let mut argmax_i: Vec<i32> = Vec::new();
        let mut argmax_j: Vec<i32> = Vec::new();
        let mut argmax_vals: Vec<f64> = Vec::new();
        let actions: Vec<i32> = (self.actions.start..self.actions.end).collect();
        //
        for c in 0..m {
            let matcomp = transposes.get(&actions[pi[c] as usize]).unwrap();
            let p = &matcomp.p;
            let i = &matcomp.i;
            let x = &matcomp.x;
            if p[c + 1] - p[c] > 0 {
                for r in p[c]..p[c + 1] { // for each row recorded in CSS add the transpose of the coord
                    argmax_j.push(i[r as usize]);
                    argmax_i.push(c as i32);
                    argmax_vals.push(x[r as usize]);
                }
            }
        }

        //println!("i {:?}", argmax_i);
        //println!("j {:?}", argmax_j);
        //println!("vals: {:?}", argmax_vals);
        let nnz = argmax_vals.len();
        let argmaxT = create_sparse_matrix(
            m as i32,
            n as i32,
            &mut argmax_i[..],
            &mut argmax_j[..],
            &mut argmax_vals[..]
        );
        let argmaxA = convert_to_compressed(argmaxT);

        SparseMatrixAttr {
            m: argmaxA,
            nr: m,
            nc: n,
            nnz: nnz
        }
    }

    /// This fn is a special function for determining in the largest cost for the initial policy.
    /// It does not consider task rewards because they are positive. Therefore it is only the agent
    /// vector rewards that is the input
    /// number of rows will be states (s in S) * num objectives (agents + tasks)
    fn incremental_construct_argmax_Rvector_hdd(
        pi: &[f64],
        a: i32,
        nr: usize,
        act_start: i32,
        act_end: i32,
        rmatrices: &FastHM<i32, Vec<f64>>
    ) -> Vec<f64> {
        let actions: Vec<i32> = (act_start..act_end).collect();
        let mut R: Vec<f64> = vec![0f64; nr];
        for r in 0..nr {
            let action = actions[pi[r] as usize];
            let row = rmatrices.get(&action).unwrap();
            R[r] = row[a as usize * nr + r];
        }
        R
    }

    fn sparse_policy_value(
        &self,
        eps: f64,
        pi: &[f64],
        rx: &Receiver<ChannelMetaData>,
        i: i32,
        j: i32
    ) ->(f64, f64) {
        let ns = *self.state_dims.get(&(i, j)).unwrap();
        let nobjs = self.num_agents + self.num_tasks;
        let nsprime = *self.value_vec_dims.get(&(i, j)).unwrap(); // number of total states

        let mut cs_matrices: Vec<_> = Vec::new();
        let mut rewards_map: FastHM<i32, Vec<f64>> = FastHM::new();

        for action in self.actions.start..self.actions.end {
            // for each of the action receive the Sparse matrix that is required
            // for processing
            match rx.recv() {
                Ok(mat) => {
                    rewards_map.insert(action, mat.R);
                    //coo_map.insert(action, mat.A);
                    //sparse_matrices.push(mat);
                    cs_matrices.push(SparseMatrixAttr {
                        m: sparse_to_cs(&mat.A),
                        nr: mat.A.nr as usize,
                        nc: mat.A.nc as usize,
                        nnz: mat.A.nz as usize
                    });
                }
                Err(_) => {}
            }
        }

        let argmaxP = self.incremental_construct_argmax_SpPMatrix_hdd(
            &pi[..],
            i,
            j,
            &cs_matrices[..]
        );

        let argmaxR = Self::incremental_construct_argmax_Rmatrix_hdd(
            &pi[..],
            ns,
            nobjs,
            self.actions.start,
            self.actions.end,
            &rewards_map
        );

        self.value_for_policy_sparse(
            eps,
            nobjs,
            &ns,
            &nsprime,
            &argmaxP,
            &argmaxR[..],
            i,
            j
        )
    }

    fn runner_extreme_points(
        &self,
        eps: f64,
        na: usize,
        w: &[f64],
        queue_load_thresh: i32,
        number_extreme_points: usize,
        num_threads: usize
    ) -> (Vec<FastHM<(i32, i32), Vec<f64>>>, FastHM<usize, Vec<f64>>) {
        let (tx, rx): (SyncSender<ChannelMetaData>, Receiver<ChannelMetaData>) =
            sync_channel(queue_load_thresh as usize * na);
        fsend(
            &(self.num_tasks as i32),
            &(self.num_agents as i32),
            &tx,
            queue_load_thresh,
            self.actions.start, self.actions.end
        );
        //let (mu, r) = self.sparse_value_iter_hdd(rx, eps, na, w);
        self.calculate_extreme_points(rx, num_threads, eps, na, w, number_extreme_points)
    }

    fn runner_sparse_value_iteration(
        &self,
        eps: f64,
        na: usize,
        w: &[f64],
        queue_load_thresh: i32
    ) -> (FastHM<(i32, i32), Vec<f64>>, Vec<f64>) {
        let (tx, rx): (SyncSender<ChannelMetaData>, Receiver<ChannelMetaData>) =
            sync_channel(queue_load_thresh as usize * na);
        fsend(
            &(self.num_tasks as i32),
            &(self.num_agents as i32),
            &tx,
            queue_load_thresh,
            self.actions.start, self.actions.end
        );
        //let (mu, r) = self.sparse_value_iter_hdd(rx, eps, na, w);
        self.sparse_value_iter_hdd(rx, eps, na, w)
    }

    fn runner_policy_value(
        &self,
        eps: f64,
        na: usize,
        pi: &[f64],
        agent: i32,
        task: i32
    ) -> (f64, f64) {
        // load in the sparse matrix for the
        let (tx, rx): (SyncSender<ChannelMetaData>, Receiver<ChannelMetaData>) =
            sync_channel(na);

        fsend_task_agent(
            task,
            agent,
            &tx,
            self.actions.start,
            self.actions.end,
            self.num_tasks + self.num_agents
        );

        self.sparse_policy_value(
            eps,
            pi,
            &rx,
            agent,
            task
        )
    }

    fn incremental_construct_argmax_Rmatrix_hdd(
        pi: &[f64],
        nr: usize,
        nc: usize,
        act_start: i32,
        act_end: i32,
        rmatrices: &FastHM<i32, Vec<f64>>
    ) -> Vec<f64> {
        let actions: Vec<i32> = (act_start..act_end).collect();
        let mut R: Vec<f64> = vec![0f64; nr * nc];
        for r in 0..nr {
            let action = actions[pi[r] as usize];
            let row = rmatrices.get(&action).unwrap();
            for c in 0..nc {
                R[c * nr + r] = row[c * nr + r];
            }
        }
        R
    }

    fn update_switch_values_mutex(
        &self,
        a: i32,
        t: i32,
        x: &mut [f64],
        X: &mut [f64],
        Xnew: &mut [f64],
        next_agent_value_cache: &ValueCache,
        next_task_value_cache: &ValueCache,
        ns: usize
    ) {
        match self.switch_state_mapping.get(&(a, t)) {
            Some(map) => {
                match map.get(&SwitchTypeCache::NextTask) {
                    Some(v) => {
                        for y in v.iter() {
                            match next_task_value_cache.value {
                                Some(val) => { x[y.dynamic_link_idx] = val }
                                None => { }
                            }
                            match &next_task_value_cache.obj_values {
                                Some(val) => {
                                    for k in 0..self.num_agents + self.num_tasks {
                                        X[k * ns + y.dynamic_link_idx] = val[k];
                                        Xnew[k * ns + y.dynamic_link_idx] = val[k];
                                    }
                                }
                                None => { }
                            }
                        }
                    }
                    None => { }
                }
            }
            None => { }
        }
        match self.switch_state_mapping.get(&(a, t)) {
            Some(map) => {
                match map.get(&SwitchTypeCache::NextAgent) {
                    Some(v) => {
                        for y in v.iter() {
                            match next_agent_value_cache.value {
                                Some(val) => { x[y.dynamic_link_idx] = val }
                                None => { }
                            }
                            match &next_agent_value_cache.obj_values {
                                Some(val) => {
                                    for k in 0..self.num_tasks + self.num_agents {
                                        X[k * ns + y.dynamic_link_idx] = val[k];
                                        Xnew[k * ns + y.dynamic_link_idx] = val[k];
                                    }
                                }
                                None => { }
                            }
                        }
                    }
                    None => { }
                }
            }
            None => { }
        }
    }

    fn subset_value_iteration(
        &self,
        e: usize,
        w: &[f64],
        na: usize,
        Pi: Arc<Mutex<&mut Vec<FastHM<(i32, i32), Vec<f64>>>>>,
        cache: Arc<Mutex<&mut Vec<FastHM<SwitchTypeCache, ValueCache>>>>,
        matrices: Arc<FastHM<i32, COO>>,
        rewards: Arc<FastHM<i32, Vec<f64>>>,
        eps: &f64,
        i: i32,
        j: i32
    ) {
        let mut cs_matrices: Vec<SparseMatrixAttr> = Vec::new();
        for action in self.actions.start..self.actions.end {
            let coo_matrix = matrices.get(&action).unwrap();
            cs_matrices.push(SparseMatrixAttr {
                m: sparse_to_cs(&coo_matrix),
                nr: coo_matrix.nr as usize,
                nc: coo_matrix.nc as usize,
                nnz: coo_matrix.nz as usize
            });
        }
        let ns = *self.state_dims.get(&(i, j)).unwrap(); // number of unmodified states
        let nsprime = *self.value_vec_dims.get(&(i, j)).unwrap(); // number of total states
        //println!("Agent: {}, Task: {}, |S|: {}, |S'|: {}", i, j, ns, nsprime);
        // including switch transition states (links the product matrices in sequence)
        // x, x:
        let nobjs = self.num_agents + self.num_tasks;
        let mut pi: Vec<f64> = vec![-1.0; ns];
        let mut pi_new: Vec<f64> = vec![-1.0; ns];
        let mut x = vec![0f64; nsprime];
        let mut xnew = vec![0f64; nsprime]; // new value vector for agent-task pair
        let mut xtemp = vec![0f64; nsprime]; // a temporary vector used for copying
        let mut X: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mut Xnew: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mut Xtemp: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mut epsold: Vec<f64> = vec![0f64; nsprime * nobjs];
        let mut inf_indices: Vec<f64>;// = Vec::new();
        let mut inf_indices_old: Vec<f64> = Vec::new();
        let mut unstable_count: i32 = 0;

        {
            let agent_cache = cache.lock().unwrap();
            let next_agent_cache = agent_cache[e].get(&SwitchTypeCache::NextAgent).unwrap();
            let next_task_cache = agent_cache[e].get(&SwitchTypeCache::NextTask).unwrap();
            self.update_switch_values_mutex(
                i,
                j,
                &mut x[..],
                &mut X[..],
                &mut Xnew[..],
                next_agent_cache,
                next_task_cache,
                nsprime);
            std::mem::drop(agent_cache);
        }

        //println!("e: {} w: {:?}\nX: {:.2?}", e, w, X);

        // make a random policy
        self.incremental_get_rand_proper_policy(
            i, j, &mut pi[..]
        );
        // construct the transition matrix generated from the random init policy
        let argmaxPinit = self.incremental_construct_argmax_SpPMatrix_hdd(
            &mut pi[..],
            i,
            j,
            &cs_matrices[..]
        );
        // construct a rewards matrix from the random init policy
        let mut b: Vec<f64> = Self::incremental_construct_argmax_Rvector_hdd(
            &pi[..],
            i,
            ns,
            self.actions.start,
            self.actions.end,
            &rewards
        );
        self.value_for_init_policy__sparse(
            &mut b[..],
            &mut x[..],
            &eps,
            &argmaxPinit
        );
        let mut epsilon: f64; // = 1.0;
        let mut policy_stable = false;
        let mut q = vec![0f64; ns * (self.actions.end - self.actions.start) as usize];
        while !policy_stable {
            policy_stable = true;
            for (ii, a) in (self.actions.start..self.actions.end).enumerate() {
                let mut vmv = vec![0f64; ns];
                sp_mv_multiply_f64(cs_matrices[ii].m, &x[..], &mut vmv);
                let mut rmv = vec![0f64; cs_matrices[ii].nr as usize];
                blas_matrix_vector_mulf64(
                    &rewards.get(&a).unwrap()[..],
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
                self.update_qmat(&mut q[..], &vmv[..], ii, na).unwrap();
            }
            // determine the maximum values for each state in the matrix of value estimates
            self.max_values(&mut xnew[..], &q[..], &mut pi_new[..], ns, na as usize);

            copy(&xnew[..], &mut xtemp[..], nsprime as i32);
            // copy the new value vector to calculate epsilon
            add_vecs(&x[..], &mut xnew[..], ns as i32, -1.0);
            self.update_policy(&xnew[..], &eps, &mut pi[..], &pi_new[..], ns, &mut policy_stable);
            copy(&xtemp[..], &mut x[..], ns as i32);
        }

        let mdp_init = *self.get_init_state(i, j);
        {
            let mut agent_cache = cache.lock().unwrap();
            if i == 0 {
                // Cache storing all quantities for the next task
                agent_cache[e].get_mut(&SwitchTypeCache::NextTask).unwrap().set_value_vec(x[mdp_init]);
            } else {
                // Cache storing all quantities for next agent
                agent_cache[e].get_mut(&SwitchTypeCache::NextAgent).unwrap().set_value_vec(x[mdp_init]);
            }
            std::mem::drop(agent_cache);
        }

        // construct the argmax matrix to calculate the objective values
        let argmaxP = self.incremental_construct_argmax_SpPMatrix_hdd(
            &pi[..],
            i,
            j,
            &cs_matrices[..]
        );
        //println!("a: {}, t: {}", i, j);
        //print_matrix(argmaxP.m);

        let argmaxR = Self::incremental_construct_argmax_Rmatrix_hdd(
            &pi[..],
            ns,
            nobjs,
            self.actions.start,
            self.actions.end,
            &rewards
        );

        epsilon = 1.0;
        let mut epsilon_old: f64 = 1.0;
        while epsilon > *eps && unstable_count < UNSTABLE_POLICY {
            for k in 0..nobjs {
                let mut vobjvec = vec![0f64; ns];
                sp_mv_multiply_f64(argmaxP.m, &X[k*nsprime..(k+1)*nsprime], &mut vobjvec[..]);
                add_vecs(&argmaxR[k*ns..(k+1)*ns], &mut vobjvec[..], ns as i32, 1.0);
                copy(&vobjvec[..], &mut Xnew[k*nsprime..(k+1)*nsprime], ns as i32);
            }
            // determine the difference between X, Xnew
            let obj_len = (nsprime * (self.num_agents + self.num_tasks)) as i32;
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
            //println!("X: {:.2?}", X);
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
        for k in 0..self.num_agents + self.num_tasks {
            objvals.push(X[k * nsprime + mdp_init]);
        }

        let mut agent_cache = cache.lock().unwrap();
        if i == 0 {
            agent_cache[e].get_mut(&SwitchTypeCache::NextTask).unwrap().set_objective_vec(&objvals[..], &(self.num_agents + self.num_tasks));
        } else {
            agent_cache[e].get_mut(&SwitchTypeCache::NextAgent).unwrap().set_objective_vec(&objvals[..], &(self.num_agents + self.num_tasks));
        }
        //println!("X final [{}]\n{:.2?}", e, X);
        println!("agent cache [{}]:{:.2?}", e, agent_cache[e]);
        let mut mu = Pi.lock().unwrap();
        mu[e].insert((i, j), pi);

    }

    fn incremental_proper_policy(
        &mut self,
        mdp: &mut MDP<(i32, i32), Vec<(i32, i32)>, Vec<((i32, i32), f64)>, [f64; 2]>
    ) {
        let mut stack: Vec<(i32, i32)> = Vec::new();
        let mut proper_policies: FastHM<usize, HashSet<i32>> = FastHM::new();
        let agent = mdp.get_agent().unwrap();
        let task = mdp.get_task().unwrap();
        // starting with the initial state
        //
        // get the done states of the MDP, these are the ones we know will end in rewards
        // satisfying the stochastic shortest path
        let dones = mdp.get_labels(MDPLabel::Done);
        let mut F = dones
            .iter()
            .chain(mdp.get_labels(MDPLabel::Sw).iter())
            .map(|x| *x)
            .collect();
        //
        // this algorithm needs to be a little bit different, we need to start with
        // of the final states and then work backwards
        //
        let mut visited: Vec<bool> = vec![false; mdp.states.len()];
        stack.append(&mut F);
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
        self.proper_policies.insert((agent, task), proper_policies);
    }
}

pub enum TaskAgentOrder {
    Forward,
    Reverse
}

fn fsend(
    num_tasks: &i32,
    num_agents: &i32,
    tx: &SyncSender<ChannelMetaData>,
    queue_load_thresh: i32,
    start_act: i32,
    end_act: i32
) {
    let agents = *num_agents;
    let tasks = *num_tasks;
    let nobjs = agents + tasks;
    let thread_worker = tx.clone();
    thread::spawn(
        move || {
            let mut count = 0;
            let mut queue: VecDeque<ChannelMetaData> = VecDeque::new();
            for t in (0..tasks).rev() {
                for a in (0..agents).rev() {
                    for action in start_act..end_act {
                        let filename = &*format!("mat_{}_{}_{}.yml", a, t, action);
                        let S = COO::read_matrix_from_file(filename);
                        let rfilename = &*format!("r_{}_{}_{}.txt", a, t, action);
                        let R = read_rewards_matrix(rfilename, S.nr as usize, nobjs as usize);
                        //println!("COO({},{})\ni:{:?}\nj:{:?}\nx{:?}",a, t, S.i, S.j, S.x);
                        queue.push_back(ChannelMetaData{ A: S, R, a, t , act: action});
                        count += 1;
                        if count >= queue_load_thresh {
                            // empty the queue
                            while !queue.is_empty() {
                                thread_worker.send(queue.pop_front().unwrap()).unwrap();
                            }
                        }
                    }
                }
            }
        }
    );
}

fn fsend_task_agent(
    task: i32,
    agent: i32,
    tx: &SyncSender<ChannelMetaData>,
    start_act: i32,
    end_act: i32,
    nobjs: usize
) {
    let thread_worker = tx.clone();
    thread::spawn(
        move || {
            let mut queue: VecDeque<ChannelMetaData> = VecDeque::new();
            for action in start_act..end_act {
                let filename = &*format!("mat_{}_{}_{}.yml", agent, task, action);
                let S = COO::read_matrix_from_file(filename);
                let rfilename = &*format!("r_{}_{}_{}.txt", agent, task, action);
                let R = read_rewards_matrix(
                    rfilename,
                    S.nr as usize,
                    nobjs
                );
                queue.push_back(ChannelMetaData{ A: S, R, a: agent, t: task, act: action});
                while !queue.is_empty() {
                    thread_worker.send(queue.pop_front().unwrap()).unwrap();
                }
            }
        }
    );
}