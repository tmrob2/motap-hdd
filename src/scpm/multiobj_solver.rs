#![allow(non_snake_case)]

use std::iter::FromIterator;
use std::time::Instant;
use hashbrown::{HashMap as FastHM, HashSet as FastSet, HashSet};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use pyo3::prelude::*;
use crate::algorithm::motap_solver::{IMOVISolver, MergeState, MultiObjSolver};
use crate::{COO, deconstruct, sparse_to_cs, SparseMatrixComponents, transpose, blas_dot_product};
use crate::scpm::definition::{SCPM, SwitchMappings, SwitchTypeCache, TaskAgentStateActionPair};
use crate::scpm::solver_utils::Mantissa;
use crate::algorithm::lp_solver::LPSolver;
use crate::utils::number_fmts::val_or_zero_one;

impl MultiObjSolver for SCPM {
    fn imovi_hdd_multi_object_solver(&self, eps: f64, t: &[f64], cost_step: f64, prob_step: f64)
        -> (Vec<FastHM<(i32, i32), Vec<f64>>>, FastHM<usize, Vec<f64>>) {
        let t1 = Instant::now();
        let mut hullset: FastHM<usize, Vec<f64>> = FastHM::new();
        let mut weights : FastHM<usize, Vec<f64>> = FastHM::new();
        let mut X: FastSet<Vec<Mantissa>> = FastSet::new();
        let mut W: FastSet<Vec<Mantissa>> = FastSet::new();
        let na: usize = (self.actions.end - self.actions.start) as usize;
        let mut schedulers: Vec<FastHM<(i32, i32), Vec<f64>>> = Vec::new();
        let tot_objs: usize = self.num_agents + self.num_tasks;
        let w = vec![1. / (self.num_agents + self.num_tasks) as f64; self.num_agents + self.num_tasks];
        let (mu, r) =
            self.runner_sparse_value_iteration(
                eps,
                na as usize,
                &w[..], (1 * na) as i32);

        let wrl = blas_dot_product(&r[..], &w[..]);
        let wt = blas_dot_product(&t[..], &w[..]);
        println!("w.r_l = {:.3}, w.t = {:.3}, wrl < wt: {}\n", wrl, wt, wrl < wt);
        if w.len() <= 10 {
            println!("r: {:.3?}\n", r);
        }
        if wrl < wt {
            println!("Ran in t(s): {:?}\n", t1.elapsed().as_secs_f64());
            pyo3::prepare_freethreaded_python();
            let code = include_str!(concat!(env!("CARGO_MANIFEST_DIR"),"/quadratic_program/qp.py"));
            // Construct a weight matrix in a way that numpy can interpret which is Vec<Vec<f64>>
            let mut qp_w_input: Vec<Vec<f64>> = Vec::new();
            let mut qp_x_input: Vec<Vec<f64>> = Vec::new();
            qp_w_input.push(w.to_vec());
            qp_x_input.push(r.to_vec());
            let mut target_min: Vec<f64> = vec![0.; tot_objs];
            for i in 0..self.num_agents {
                target_min[i] = t[i] - cost_step;
            }
            for j in self.num_agents..tot_objs {
                target_min[j] = t[j] - prob_step;
            }
            let tnew = Python::with_gil(|py| -> PyResult<Vec<f64>> {
                let qp = PyModule::from_code(py, code, "", "")?;
                let result: Vec<f64> = qp.getattr("quadprog_wrapper")?
                    .call1((qp_w_input, qp_x_input, 1, tot_objs, t.to_vec(), target_min.to_vec()))?
                    .extract()?;
                Ok(result)
            });
            match tnew {
                Ok(_) => {}
                Err(_) => { println!("tnew: {:?}\n", tnew); }
            }

            return (schedulers, hullset)
        }
        X.insert(r.iter()
            .cloned()
            .map(|f| Mantissa::new(f))
            .collect::<Vec<Mantissa>>()
        );
        W.insert(w
            .iter()
            .cloned()
            .map(|f| Mantissa::new(f))
            .collect::<Vec<Mantissa>>()
        );
        hullset.insert(0, r.to_vec());
        weights.insert(0, w.to_vec());
        schedulers.push(mu);

        let mut lpvalid = true;
        // Once the extreme points are calculated then we can calculate the first separating
        // hyperplane
        let mut w: Vec<f64> = vec![0.; tot_objs];
        let mut count: usize = 1;
        while lpvalid {
            let gurobi_result = self.gurobi_solver(&hullset, &t[..], &tot_objs);
            if w.len() <= 10 {
                println!("gurobi result: {:.3?}", gurobi_result);
            }
            match gurobi_result {
                //Ok(sol) => {
                Some(sol) => {
                    // construct the new w based on the values from lp solution (if it exists)
                    for (ix, val) in sol.iter().enumerate() {
                        if ix < tot_objs {
                            w[ix] = val_or_zero_one(val);
                        }
                    }

                    let new_w = w
                        .iter()
                        .clone()
                        .map(|f| Mantissa::new(*f))
                        .collect::<Vec<Mantissa>>();

                    match W.contains(&new_w) {
                        true => {
                            println!("All points discovered");
                            lpvalid = false;
                        }
                        false => {
                            // calculate the new expected weighted cost based on w
                            let (mu, r) =
                                self.runner_sparse_value_iteration(
                                    eps,
                                    na as usize,
                                    &w[..], (2 * na) as i32);

                            let wrl = blas_dot_product(&r[..], &w[..]);
                            let wt = blas_dot_product(&t[..], &w[..]);
                            if w.len() <= 10 {
                                println!("w: {:3.3?},\nr: {:.1?}", w, r);
                            }
                            println!("w.r_l = {:.3}, w.t = {:.3}, wrl < wt: {}", wrl, wt, wrl < wt);
                            if wrl < wt {
                                println!("Computing new point tnew\n");
                                println!("Ran in t(s): {:?}\n", t1.elapsed().as_secs_f64());
                                // compute the nearest point from the target vector which achieves allows the
                                // algorithm to continue
                                pyo3::prepare_freethreaded_python();
                                let code = include_str!(concat!(env!("CARGO_MANIFEST_DIR"),"/quadratic_program/qp.py"));
                                // Construct a weight matrix in a way that numpy can interpret which is Vec<Vec<f64>>
                                let mut qp_w_input: Vec<Vec<f64>> = Vec::new();
                                let mut qp_x_input: Vec<Vec<f64>> = Vec::new();
                                for k in 0..weights.len() {
                                    qp_w_input.push(weights.get(&k).unwrap().to_vec());
                                    qp_x_input.push(hullset.get(&k).unwrap().to_vec());
                                }
                                qp_w_input.push(w.to_vec()); // push the newly found weight
                                qp_x_input.push(r.to_vec()); // push the newly found expected cost
                                let mut target_min: Vec<f64> = vec![0.; tot_objs];
                                for i in 0..self.num_agents {
                                    target_min[i] = t[i] - cost_step;
                                }
                                for j in self.num_agents..tot_objs {
                                    target_min[j] = t[j] - prob_step;
                                }
                                let tnew = Python::with_gil(|py| -> PyResult<Vec<f64>> {
                                    let qp = PyModule::from_code(py, code, "", "")?;
                                    let result: Vec<f64> = qp.getattr("quadprog_wrapper")?
                                        .call1((qp_w_input, qp_x_input, weights.len() + 1, tot_objs, t.to_vec(), target_min.to_vec()))?
                                        .extract()?;
                                    Ok(result)
                                });
                                println!("tnew: {:?}\n", tnew);
                                return (schedulers, hullset)
                            }
                            // Insert the new solution
                            schedulers.push(mu);
                            hullset.insert(count, r);
                            // create a copy of the weight vector and insert it into the set of values
                            W.insert(new_w);
                            weights.insert(count, w.to_vec());
                            count += 1;
                        }
                    }
                }
                None => {
                    println!("infeasible");
                    // the LP has finished and there are no more points which can be added to the
                    // the polytope
                    lpvalid = false;
                }
            }
        }
        if w.len() <= 10 {
            for ix in 0..hullset.len() {
                println!("w: {:.3?},\n\tx: {:.1?}", weights.get(&ix).unwrap(), hullset.get(&ix).unwrap());
            }
        }
        println!("Ran in t(s): {:?}", t1.elapsed().as_secs_f64());

        (schedulers, hullset)
    }

    fn merge_schedulers_hdd2(
        simple_schedulers: &[FastHM<(i32, i32), Vec<f64>>],
        weight_vector: &[f64],
        start_act: i32,
        end_act: i32,
        num_agents: usize,
        num_tasks: usize,
        initial_state: usize,
        switch_mapping: &FastHM<(i32, i32), FastHM<SwitchTypeCache, Vec<SwitchMappings>>>
        ) -> (FastSet<MergeState>, FastHM<MergeState, FastHM<Vec<i32>, (MergeState, i32, f64)>>){
        //let mut stack: Vec<MergeState> = Vec::new();
        let mut V: FastSet<MergeState> = FastSet::new();
        let mut E: FastHM<MergeState, FastHM<Vec<i32>, (MergeState, i32, f64)>> = FastHM::new();
        let actions: Vec<i32> = (start_act..end_act).collect();
        println!("actions: {:?}", actions);
        // first construct a Hashmap comprising all of the deconstructed Sparse BLAS matrices
        //for l in 0..simple_schedulers.len() {
        //    println!("simple schedulers\n:{:?}", simple_schedulers[l].get(&(0, 0)));
        //}

        let mut decomposed: FastHM<((i32, i32), i32), SparseMatrixComponents> = FastHM::new();

        // 1. load in the transition matrices for agent: i and task: j
        //
        // 2. then proceed to find all of the scheduled state-actions relevant to the those transition
        // matrices
        //
        // 3. Once all of these transitions have been exhausted drop this matrix and move onto the next
        // matrix. It may be that the agent immediately transitions to the next agent
        //
        // 4. repeat


        let mut stacks: FastHM<(i32, i32), Vec<MergeState>> = FastHM::new();
        for a in 0..num_agents {
            for t in 0..num_tasks {
                stacks.insert((a as i32, t as i32), Vec::new());
            }
        }

        stacks.get_mut(&(0, 0)).unwrap().push(MergeState {
            state: initial_state,
            agent: 0,
            task: 0,
            C: (0..simple_schedulers.len() as i32).collect::<Vec<i32>>(),
            initial: true
        });

        let bar = ProgressBar::new((num_agents * num_tasks) as u64);
        bar.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.red/blue} {pos:>7}/{len:7} {msg}")
            .progress_chars("##-"));
        // if the task is the last task then we need to know which agent actually gets the task
        for task in 0..num_tasks {
            for agent in 0..num_agents {
                bar.inc(1);
                for a in start_act..end_act {
                    let filename = &*format!("mat_{}_{}_{}.yml", agent, task, a);
                    let S = COO::read_matrix_from_file(filename);
                    let P = sparse_to_cs(&S);
                    // Now we actually require the transpose of the BLAS matrix as we
                    // want to know for each row which cols result in non-zero value P(s, a, s') > 0
                    let PT = transpose(P, S.nz as i32);
                    // We also need to swap the col count with the row count as the matrix is now
                    // on its side
                    decomposed.insert(((agent as i32, task as i32), a),
                                      deconstruct(PT, S.nz as usize, S.nr as usize));
                };
                while !stacks.get(&(agent as i32, task as i32)).unwrap().is_empty() {
                    let new_s = stacks.get_mut(&(agent as i32, task as i32)).unwrap().pop().unwrap();
                    if !V.contains(&new_s) {
                        for a in start_act..end_act {
                            match decomposed.get(&((agent as i32, task as i32), a)) {
                                Some(sparse_components) => {
                                    if sparse_components.p[new_s.state + 1] - sparse_components.p[new_s.state] > 0 {
                                        // then there is at least one element in this column
                                        let mut Cprime: Vec<i32> = vec![];
                                        for ii in new_s.C.iter() {
                                            let mu_s = simple_schedulers[*ii as usize]
                                                .get(&(new_s.agent as i32, new_s.task as i32))
                                                .expect(
                                                    format!("Couldn't retrieve mu for A:{}, T: {}, l: {}",
                                                            new_s.agent, new_s.task, ii).as_str()
                                                )[new_s.state];
                                            if actions[mu_s as usize] == a {
                                                //println!("scheduler: {} is used in state: {} => a: {}", ii, c, a);
                                                Cprime.push(*ii);
                                            }
                                        }
                                        for r in sparse_components.p[new_s.state]..sparse_components.p[new_s.state + 1] { // get the non zero matrix elements
                                            // so in this context c is s, and r is s'
                                            // get the action for scheduler i, in the set C, for state c
                                            //println!("A: {}, T: {}, s: {} -> s': {}", new_s.agent, new_s.task, new_s.state, sparse_components.i[r as usize]);
                                            let switches = switch_mapping.get(&(new_s.agent as i32, new_s.task as i32));
                                            let mut new_agent: usize = new_s.agent;
                                            let mut new_task: usize = new_s.task;
                                            let mut new_state: usize = sparse_components.i[r as usize] as usize;
                                            let mut initial: bool = false;
                                            match switches {
                                                Some(v) => {
                                                    match v.get(&SwitchTypeCache::NextAgent) {
                                                        Some(sw_next_agent) => {
                                                            if !sw_next_agent.is_empty() {
                                                                if sw_next_agent[0].dynamic_link_idx == sparse_components.i[r as usize] as usize {
                                                                    new_agent = new_s.agent + 1;
                                                                    new_task = new_s.task;
                                                                    new_state = sw_next_agent[0].true_idx;
                                                                    initial = true;
                                                                }
                                                            }
                                                        }
                                                        None => { }
                                                    }
                                                    match v.get(&SwitchTypeCache::NextTask) {
                                                        Some(sw_next_task) => {
                                                            if !sw_next_task.is_empty() {
                                                                if sw_next_task[0].dynamic_link_idx == sparse_components.i[r as usize] as usize {
                                                                    new_agent = 0;
                                                                    new_task = new_s.task + 1;
                                                                    new_state = sw_next_task[0].true_idx;
                                                                    initial = true
                                                                }
                                                            }
                                                        }
                                                        None => { }
                                                    }
                                                }
                                                None => { }
                                            }
                                            //println!("A: {}, T: {}, State: {}", new_s.agent, new_s.task, new_s.state);
                                            let denom = new_s.C.iter().fold(0.0, |acc, k| acc + weight_vector[*k as usize]);
                                            //println!("C: {:?}, val: {}", new_s.C, denom);
                                            let numer = Cprime.iter().fold(0.0, |acc, k| acc + weight_vector[*k as usize]);
                                            //println!("C': {:?}, A:{}, T: {}, val: {}", Cprime, new_agent, new_task, numer);
                                            let p = if denom > 0f64 { numer / denom } else { 0. };
                                            //println!("p: {}", p);
                                            if p > 0. {
                                                if !Cprime.is_empty() {
                                                    stacks.get_mut(&(new_agent as i32, new_task as i32)).unwrap().push(MergeState {
                                                        state: new_state,
                                                        agent: new_agent,
                                                        task: new_task,
                                                        C: Cprime.to_vec(),
                                                        initial
                                                    });
                                                }
                                                match E.get_mut(&new_s) {
                                                    None => {
                                                        E.insert(
                                                            new_s.clone(),
                                                            FastHM::from_iter(vec![(Cprime.to_vec(), (MergeState {
                                                                state: new_state,
                                                                agent: new_agent,
                                                                task: new_task,
                                                                C: Cprime.to_vec(),
                                                                initial
                                                            }, a, p))]
                                                            ));
                                                    }
                                                    Some(v) => {
                                                        v.insert(
                                                            Cprime.to_vec(),
                                                            (MergeState {
                                                                state: new_state,
                                                                agent: new_agent,
                                                                task: new_task,
                                                                C: Cprime.to_vec(),
                                                                initial
                                                            }, a, p)
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                        V.insert(new_s);
                    }
                }
            }
        }
        (V, E)
    }

    /// nt: number of tasks
    /// na: number of agents
    /// V: vertices of randomised scheduler
    /// E: edged of randomised scheduler
    fn choose_and_decomp_scheduler_hdd(
        V: &HashSet<MergeState>,
        E: &FastHM<MergeState, FastHM<Vec<i32>, (MergeState, i32, f64)>>,
        nt: usize,
        na: usize,
        init_state: usize
    ) -> FastHM<(i32, i32), HashSet<TaskAgentStateActionPair>> {
        let mut schedulers: FastHM<(i32, i32), HashSet<TaskAgentStateActionPair>> = FastHM::new();

        for a in 0..na {
            for t in 0..nt {
                schedulers.insert((a as i32, t as i32), HashSet::new());
            }
        }

        // for each task create a stack
        let mut stack: Vec<MergeState> = Vec::new();
        // get the merge state at the initial task agent pair
        let v0= V.iter()
            .find(|m| m.task == 0 &&
                m.agent == 0 &&
                m.state == init_state
            ).unwrap();
        stack.push(v0.clone());

        while !stack.is_empty() {
            let snew = stack.pop().unwrap();
            // lookup the current state in the Edge list from the output randomised scheduler
            // here, we are presented with a randomised choice
            let mut choices: Vec<(&[i32], &f64)> = Vec::new();
            match E.get(&snew) {
                Some(x) => {
                    for (C, (_, _, p)) in x {
                        // we need to make a choice
                        choices.push((&C[..], p));
                    }
                }
                None => { panic!("s: {:?} not found", snew) }
            }

            let choice = choices.choose_weighted(&mut thread_rng(), |item| item.1).unwrap().0;
            let (sprime, action, _p) =
                E.get(&snew).unwrap().get(choice).unwrap();
            // insert this new state into the stack if it has not already been visited
            match schedulers.get_mut(&(snew.agent as i32, snew.task as i32)) {
                None => {
                    panic!("Not possible, all values should be initialised")
                }
                Some(v) => {
                    if v.contains(&TaskAgentStateActionPair {
                        s: snew.state,
                        s_a: snew.agent,
                        s_t: snew.task,
                        action: *action,
                        sprime: sprime.state,
                        sprime_a: sprime.agent,
                        sprime_t: sprime.task
                    }) {
                    } else {
                        v.insert(TaskAgentStateActionPair {
                            s: snew.state,
                            s_a: snew.agent,
                            s_t: snew.task,
                            action: *action,
                            sprime: sprime.state,
                            sprime_a: sprime.agent,
                            sprime_t: sprime.task
                        });
                        stack.push(sprime.clone());
                    }
                }
            }
        }
        schedulers
    }
}