#![allow(non_snake_case)]

use hashbrown::{HashMap as FastHM, HashSet as FastSet};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::Receiver;
use serde::{Serialize, Deserialize};
use crate::{ChannelMetaData, COO};
use crate::definition::MDP;
use crate::scpm::definition::{MatrixAttr, SparseMatrixAttr, SwitchMappings,
                              SwitchTypeCache, TaskAgentStateActionPair, ValueCache};

#[derive(Debug)]
pub struct StateMap {
    pub idx: usize,
    pub agent: usize,
    pub task: usize,
    pub state: (i32, i32)
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Default, Serialize, Deserialize)]
pub struct MergeState {
    pub state: usize,
    pub agent: usize,
    pub task: usize,
    pub C: Vec<i32>,
    pub initial: bool
}

impl MergeState {
    pub fn new(state: usize, agent: usize, task: usize, C: Vec<i32>) -> Self {
        MergeState {
            state,
            agent,
            task,
            C,
            initial: false
        }
    }

    pub fn set_initial(mut m: Self) -> Self {
        m.initial = true;
        m
    }
}

pub trait MultiObjSolver {
    fn imovi_hdd_multi_object_solver(
        &self,
        eps: f64,
        t: &[f64]) -> (Vec<FastHM<(i32, i32), Vec<f64>>>, FastHM<usize, Vec<f64>>);

    fn merge_schedulers_hdd2(
        simple_schedulers: &[FastHM<(i32, i32), Vec<f64>>],
        weight_vector: &[f64],
        start_act: i32,
        end_act: i32,
        num_agents: usize,
        num_tasks: usize,
        initial_state: usize,
        switch_mapping: &FastHM<(i32, i32), FastHM<SwitchTypeCache, Vec<SwitchMappings>>>
    ) -> (FastSet<MergeState>, FastHM<MergeState, FastHM<Vec<i32>, (MergeState, i32, f64)>>);

    fn choose_and_decomp_scheduler_hdd(
        V: &hashbrown::HashSet<MergeState>,
        E: &FastHM<MergeState, FastHM<Vec<i32>, (MergeState, i32, f64)>>,
        nt: usize,
        na: usize,
        init_state: usize
    ) -> FastHM<(i32, i32), hashbrown::HashSet<TaskAgentStateActionPair>>;
}

pub trait IMOVISolver {
    fn sparse_value_iter_hdd(&self, rx: Receiver<ChannelMetaData>, eps: f64, na: usize, w: &[f64])
        -> (FastHM<(i32, i32), Vec<f64>>, Vec<f64>);

    fn incremental_construct_argmax_SpPMatrix_hdd(
        &self,
        pi: &[f64],
        a: i32,
        t: i32,
        matrices: &[SparseMatrixAttr]
    ) -> SparseMatrixAttr;

    fn runner_sparse_value_iteration(
        &self,
        eps: f64,
        na: usize,
        w: &[f64],
        queue_load_thresh: i32
    ) -> (FastHM<(i32, i32), Vec<f64>>, Vec<f64>);

    fn sparse_policy_value(
        &self,
        eps: f64,
        pi: &[f64],
        rx: &Receiver<ChannelMetaData>,
        agent: i32,
        task: i32
    ) ->(f64, f64);

    fn runner_policy_value(
        &self,
        eps: f64,
        na: usize,
        pi: &[f64],
        agent: i32,
        task: i32
    ) -> (f64, f64);

    fn runner_extreme_points(
        &self,
        eps: f64,
        na: usize,
        w: &[f64],
        queue_load_thresh: i32,
        number_extreme_points: usize,
        num_threads: usize
    ) -> (Vec<FastHM<(i32, i32), Vec<f64>>>, FastHM<usize, Vec<f64>>);

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
    );

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
    );

    fn calculate_extreme_points(
        &self,
        rx: Receiver<ChannelMetaData>,
        num_threads: usize,
        eps: f64,
        na: usize,
        w: &[f64],
        num_extreme_points: usize
    ) -> (Vec<FastHM<(i32, i32), Vec<f64>>>, FastHM<usize, Vec<f64>>);

    fn incremental_construct_argmax_Rvector_hdd(
        pi: &[f64],
        a: i32,
        nr: usize,
        act_start: i32,
        act_end: i32,
        rmatrices: &FastHM<i32, Vec<f64>>
    ) -> Vec<f64>;

    fn incremental_construct_argmax_Rmatrix_hdd(
        pi: &[f64],
        nr: usize,
        nc: usize,
        act_start: i32,
        act_end: i32,
        rmatrices: &FastHM<i32, Vec<f64>>
    ) -> Vec<f64>;

    fn incremental_proper_policy(
        &mut self,
        mdp: &mut MDP<(i32, i32), Vec<(i32, i32)>, Vec<((i32, i32), f64)>, [f64; 2]>
    );
}

pub trait MOVISolver {
    fn movi_value_iteration(
        &self,
        eps: f64,
        na: usize,
        w: &[f64]) -> (Vec<f64>, Vec<f64>);

    fn movi_construct_sparsePmatrix(&mut self);

    fn movi_construct_densePmatrix(&mut self);

    fn movi_construct_Rmatrix(&mut self);

    fn movi_value_for_init_policy(
        &self,
        b: &mut [f64],
        x: &mut [f64],
        eps: &f64,
        argmaxP: &SparseMatrixAttr);

    fn movi_rand_proper_policy(&self, pi: &mut [f64]);

    fn movi_set_state_mapping(&mut self);

    fn movi_print_dense_matrices(&self);

    fn movi_print_rewards_matrices(&self, action: i32);

    fn movi_print_constructed_rewards(&self, R: &[f64], nr: usize, nobj: usize);

    fn movi_construct_argmaxPmatrix(&self, pi: &[f64]) -> SparseMatrixAttr;

    fn movi_construct_argmaxRmatrix(&self, pi: &[f64]) -> MatrixAttr;

    fn movi_construct_argmaxRvector(&self, pi: &[f64]) -> Vec<f64>;

    fn movi_gather_init_costs(&self, nobj: usize, X: &[f64], ns: usize) -> Vec<f64>;
}

pub trait SolverUtils {
    #[allow(non_snake_case)]
    fn update_qmat(
        &self,
        q: &mut [f64],
        v: &[f64],
        row: usize,
        nr: usize) -> Result<(), String>;

    fn max_values(&self, x: &mut [f64], q: &[f64], pi: &mut [f64], ns: usize, na: usize);

    fn update_policy(
        &self,
        eps: &[f64],
        thresh: &f64,
        pi: &mut [f64],
        pnew: &[f64],
        ns: usize,
        policy_stable: &mut bool);

    fn max_eps(&self, x: &[f64]) -> f64;

    #[allow(non_snake_case)]
    fn gather_init_costs(&self, X: &[f64]) -> Vec<f64>;

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
    ) -> (f64, f64);

    fn value_for_init_policy__sparse(
        &self,
        b: &mut [f64],
        x: &mut [f64],
        eps: &f64,
        argmaxP: &SparseMatrixAttr);
}