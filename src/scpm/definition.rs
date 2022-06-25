use hashbrown::{HashMap as FastHM, HashSet as FastSet};
use std::collections::HashMap;
use std::env::VarError;
use std::fs;
use rand::{prelude::{SliceRandom, IteratorRandom}, thread_rng};
use serde::{Serialize, Deserialize};
use crate::cs_di;
use crate::definition::{MDP, MDPLabel, Pred};
use crate::algorithm::motap_solver::{IMOVISolver};
use crate::algorithm::lp_solver::LPSolver;

pub type XpS =  (i32, i32);
pub type XpSprPr = (XpS, f64);

pub const UNSTABLE_POLICY: i32 = 5;

#[derive(Debug, Clone, Hash, PartialEq, std::cmp::Eq, Serialize, Deserialize)]
/// Represents the mapping from one value vector to its value in the next value vector
///
/// dynamic_link_idx: the state of the current value vector
///
/// true_idx: the index of the switch state in the previous value vector which will
/// have just been calculated, i.e. a dynamic link references a true index
pub struct SwitchMappings {
    pub dynamic_link_idx: usize,
    pub true_idx: usize
}

#[derive(Eq, PartialEq, Hash)]
pub struct MOVIReverseStateMap {
    pub agent: usize,
    pub task: usize,
    pub state: (i32, i32)
}

#[derive(Debug, Hash, Eq, PartialEq, Serialize)]
pub struct TaskAgentStateActionPair {
    pub s: usize,
    pub s_a: usize,
    pub s_t: usize,
    pub action: i32,
    pub sprime: usize,
    pub sprime_a: usize,
    pub sprime_t: usize
}


#[derive(Debug)]
pub struct MatrixAttr {
    pub m: Vec<f64>, // matrix
    pub nr: usize, // number of rows
    pub nc: usize // number of columns
}

pub struct SparseMatrixAttr {
    pub m: *mut cs_di, // matrix handle
    pub nr: usize, // number of rows
    pub nc: usize, // number of columns
    pub nnz: usize // the number of non-zero elements
}

#[derive(Debug, Clone)]
pub struct ValueCache {
    pub value: Option<f64>,
    pub obj_values: Option<Vec<f64>>
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub enum SwitchTypeCache {
    NextAgent,
    NextTask
}

impl ValueCache {
    pub fn new() -> ValueCache {
        ValueCache {
            value: None,
            obj_values: None
        }
    }
    /// Consumes the vector x
    pub fn set_value_vec(&mut self, x: f64) {
        self.value = Some(x);
    }

    pub fn set_objective_vec(&mut self, x: &[f64], objs: &usize) {
        match self.obj_values.as_mut() {
            None => {
                self.obj_values = Some(x.to_vec());
            }
            Some(z) => {
                for k in 0..*objs {
                    z[k] = x[k];
                }
            }
        };
    }

    pub fn get_obj_vec(&self) -> Option<&Vec<f64>> {
        self.obj_values.as_ref()
    }
}

pub enum BLASType {
    Sparse,
    Dense
}

pub enum AlgorithmType {
    MOVI,
    IMOVI
}

/// While the MDP may take on a number of forms, the SCPM expects certain characteristics and
/// therefore the types it may take on is much stricter
pub struct SCPM {
    pub mdps: Vec<MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>>,
    pub actions: std::ops::Range<i32>,
    pub states: usize,
    pub num_transitions: usize,
    pub mdp_statespace_maps: FastHM<(i32, i32), usize>, // this is the starting index for the (a,t) MDP
    pub state_dims: HashMap<(i32, i32), usize>,
    pub init_states: FastHM<(i32, i32), usize>,
    pub value_vec_dims: HashMap<(i32, i32), usize>,
    pub blas_reward_matrices: FastHM<((i32, i32), i32), MatrixAttr>,
    //pub sparse_transitions_matrix_address_values: FastHM<((i32, i32), i32), SpMatrixFmt>,
    pub initial_state: usize,
    // We want to call the all of the switch mappings from an MDP_{i, j}
    pub switch_state_mapping: FastHM<(i32, i32), FastHM<SwitchTypeCache, Vec<SwitchMappings>>>,
    pub num_agents: usize,
    pub num_tasks: usize,
    available_actions: FastHM<(i32, i32), FastHM<usize, Vec<i32>>>,
    pub proper_policies: FastHM<(i32, i32), FastHM<usize, FastSet<i32>>>,
}

impl SCPM {
    /// Just sets the most basic properties of the SCPM such as action space, number of agents, and
    /// number of tasks. On creation there are no MDPs stored and no initial state is set.
    pub fn incremental_make(
        n_actions: i32,
        na: usize,
        nt: usize
    ) -> Self {
        let scpm_home = match std::env::var("SCPM_HOME") {
            Ok(var) => {var}
            Err(_) => { panic!("No ENV variable named $SCPM_HOME. Add env variable before continuing. See docs for more information.")}
        };
        let pth = format!("{}/rewards", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => {}
            Err(_) => { println!("No dir named rewards in $SCPM_HOME.")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/transitions", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => {println!("No dir named transitions in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/schedulers", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => { println!("No dir named schedulers in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/mappings", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => { println!("No dir named mappings in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/graph", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => { println!("No dir named graph in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/switches", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => { println!("No dir named switches in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/data/solutions", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => { println!("No dir named data/solutions in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/data/metadata", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => { println!("No dir named data/metadata in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let pth = format!("{}/data/robot", scpm_home);
        match fs::remove_dir_all(pth.as_str()) {
            Ok(_) => { }
            Err(_) => { println!("No dir named data/robot in $SCPM_HOME")}
        };
        fs::create_dir_all(pth.as_str()).unwrap();
        let scpm = SCPM {
            mdps: Vec::new(), // The SCPM takes ownership of the MDPs at this point
            actions: -1..n_actions,
            states: 0,
            num_transitions: 0,
            mdp_statespace_maps: FastHM::new(),
            state_dims: HashMap::new(),
            value_vec_dims: HashMap::new(),
            init_states: FastHM::new(),
            blas_reward_matrices: FastHM::new(),
            initial_state: 0,
            switch_state_mapping: FastHM::new(),
            num_agents: na,
            num_tasks: nt,
            available_actions: FastHM::new(),
            proper_policies: Default::default()
        };

        scpm
    }

    pub fn add_mdp_to_self(
        &mut self,
        mdp: &mut MDP<(i32, i32), Vec<(i32, i32)>, Vec<((i32, i32), f64)>, [f64; 2]>,
        next_agent_idx: usize,
        next_task_idx: usize
    ) {
        self.insert_init_state(
            mdp.get_agent().unwrap(),
            mdp.get_task().unwrap(),
            *mdp.state_mapping.get(&mdp.init_state).unwrap()
        );
        self.incremental_set_state_space_size(mdp);
        self.incremental_modify_forward(mdp, next_task_idx);
        self.incremental_modify_adjacent(mdp, next_agent_idx);
        self.incremental_add_mdp_transitions(mdp);
        self.incremental_rewards_fn(mdp);
        self.incremental_set_available_actions(mdp);
        self.incremental_set_value_vec_size(mdp);
        self.incremental_proper_policy(mdp);
    }

    pub fn get_init_state(&self, a: i32, t: i32) -> &usize {
        self.init_states.get(&(a, t)).unwrap()
    }

    fn insert_init_state(&mut self, a: i32, t: i32, idx: usize) {
        self.init_states.insert((a, t), idx);
    }

    /// Must be called before the underlying MDP states spaces are modified
    pub fn set_state_space_size(&mut self) {
        for mdp in self.mdps.iter() {
            self.state_dims.insert(
                (mdp.get_agent().unwrap(), mdp.get_task().unwrap()),
                mdp.states.len()
            );
            self.states += mdp.states.len();
            self.init_states.insert(
                (mdp.get_agent().unwrap(),
                 mdp.get_task().unwrap()),
                *mdp.state_mapping.get(&mdp.init_state).unwrap()
            );
        }
        let mut base_idx: usize = 0;
        for t in 0..self.num_tasks as i32 {
            for a in 0..self.num_agents as i32 {
                // get the state length of the MDP
                let mdp = self.mdps
                    .iter()
                    .find(|m| m.get_agent().unwrap() == a && m.get_task().unwrap() == t
                    )
                    .unwrap();
                self.mdp_statespace_maps.insert((a, t), base_idx);
                base_idx += mdp.states.len();
            }
        }
    }

    pub fn incremental_set_state_space_size(
        &mut self,
        mdp: &MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>
    ) {
        self.states += mdp.states.len();
        // mapping the MDP state len before switch modifications
        // have been done
        self.state_dims.insert(
            (mdp.get_agent().unwrap(), mdp.get_task().unwrap()),
            mdp.states.len()
        );
        self.mdp_statespace_maps.insert(
            (mdp.get_agent().unwrap(), mdp.get_task().unwrap()),
            mdp.states.len()
        );
    }

    pub fn incremental_add_mdp_transitions(
        &mut self,
        mdp: &mut MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>
    ) {
        self.num_transitions += mdp.transitions.len();
    }

    /// Must be called after MDP modifications have been completed
    pub fn set_value_vec_size(&mut self) {
        for mdp in self.mdps.iter() {
            self.value_vec_dims
                .insert((mdp.get_agent().unwrap(), mdp.get_task().unwrap()), mdp.states.len());
        }
    }

    pub fn incremental_set_value_vec_size(
        &mut self,
        mdp: &mut MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>
    ) {
        self.value_vec_dims
            .insert((mdp.get_agent().unwrap(), mdp.get_task().unwrap()), mdp.states.len());
    }

    pub fn get_state_dims(&self, agent: i32, task: i32) -> usize {
        *self.state_dims.get(&(agent, task)).unwrap()
    }

    pub fn get_state_space_size(&self) -> usize {
        self.states
    }

    pub fn get_rewards_matrix(&self, agent: i32, task: i32, a: i32) -> (&[f64], usize, usize) {
        let m = self.blas_reward_matrices.get(&((agent, task), a)).unwrap();
        (m.m.as_slice(), m.nr, m.nc)
    }

    fn incremental_set_available_actions(
        &mut self,
        mdp: &MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>
    ) {
        let mut mdp_available_actions: FastHM<usize, Vec<i32>> = FastHM::new();
        let i = mdp.get_agent().unwrap();
        let j = mdp.get_task().unwrap();

        // for each state of the mdp get the available actions for each of the states
        // get the mdp first
        for state in mdp.states.iter() {
            let state_idx = *mdp.state_mapping.get(state).unwrap();
            let mut action_set: Vec<i32> = Vec::new();
            //println!("scpm action range: [{}..{}]", self.actions.start, self.actions.end);
            for action in self.actions.start..self.actions.end {
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
        self.available_actions.insert((i as i32, j as i32), mdp_available_actions);
    }

    pub fn get_available_actions(&self) -> &FastHM<(i32, i32), FastHM<usize, Vec<i32>>> {
        &self.available_actions
    }

    /// Returns the switch from to transition indices relative to state positions in the product
    /// MDP
    ///
    /// a => agent
    ///
    /// t => task
    pub fn get_switch_indices(&self, a: i32, t: i32) -> Option<&hashbrown::HashMap<SwitchTypeCache, Vec<SwitchMappings>>> {
        self.switch_state_mapping.get(&(a, t))
    }

    fn incremental_rewards_fn(
        &mut self,
        mdp: &mut MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>
    ) {
        //let i = mdp.get_agent().unwrap();
        let j = mdp.get_task().unwrap();

        let ec =  mdp.get_labels(MDPLabel::EC).to_vec();
        let accepting = mdp.get_labels(MDPLabel::Suc).to_vec();
        //println!("EC: {:?}", ec);
        //println!("acc: {:?}", accepting);
        //if i == self.num_agents as i32 - 1 && j == self.num_tasks as i32 - 1{
        if j == self.num_tasks as i32 - 1 {
            for done in ec.iter() {
                for action in self.actions.start..self.actions.end {
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
        }
        //}
        for acc in accepting.iter() {
            for action in self.actions.start..self.actions.end {
                match mdp.rewards.get_mut(&(*acc, action)) {
                    Some(r) => { r[0] = 0f64; }
                    None => { }
                }
            }
        }
    }

    #[allow(non_snake_case)]
    /// x => value vector
    ///
    /// X => Objective value vector
    ///
    /// a => agent
    ///
    /// t => task
    ///
    /// cache >= previously stored value vector, objective matrix
    pub fn update_switch_values(
        &self,
        a: i32,
        t: i32,
        x: &mut [f64],
        X: &mut [f64],
        Xnew: &mut [f64],
        cache: &FastHM<SwitchTypeCache, ValueCache>, ns: usize) {
        match cache.get(&SwitchTypeCache::NextTask) {
            Some(z) => {
                match self.switch_state_mapping.get(&(a, t)){
                    Some(map) => {
                        match map.get(&SwitchTypeCache::NextTask) {
                            Some(v) => {
                                for y in v.iter() {
                                    match z.value {
                                        Some(val) => { x[y.dynamic_link_idx] = val }
                                        None => { }
                                    }
                                    match &z.obj_values {
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
            None => {}
        }

        match cache.get(&SwitchTypeCache::NextAgent) {
            Some(z) => {
                match self.switch_state_mapping.get(&(a, t)){
                    Some(map) => {
                        match map.get(&SwitchTypeCache::NextAgent) {
                            Some(v) => {
                                for y in v.iter() {
                                    match z.value {
                                        Some(val) => { x[y.dynamic_link_idx] = val }
                                        None => { }
                                    }
                                    match &z.obj_values {
                                        Some(val) => {
                                            //println!("Setting X to values of: {:?}", val);
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
            None => {}
        }
    }

    fn update_switch_mapping(
        &mut self,
        switch_type: SwitchTypeCache,
        current_idx: usize,
        next_idx: usize,
        agent: i32,
        task: i32
    ) {
        match switch_type {
            SwitchTypeCache::NextTask => {
                match self.switch_state_mapping.get_mut(&(agent, task)) {
                    None => {
                        let mut hmap_new: FastHM<SwitchTypeCache, Vec<SwitchMappings>> = FastHM::new();
                        hmap_new.insert(SwitchTypeCache::NextTask, vec![SwitchMappings {
                            dynamic_link_idx: current_idx,
                            true_idx: next_idx
                        }]);
                        self.switch_state_mapping.insert((agent, task), hmap_new);
                    }
                    Some(x) => {
                        match x.get_mut(&SwitchTypeCache::NextTask) {
                            None => {
                                x.insert(SwitchTypeCache::NextTask, vec![SwitchMappings {
                                    dynamic_link_idx: current_idx,
                                    true_idx: next_idx
                                }]);
                            }
                            Some(v) => {
                                v.push(SwitchMappings {
                                    dynamic_link_idx: current_idx,
                                    true_idx: next_idx
                                });
                            }
                        }
                    }
                }
            }
            SwitchTypeCache::NextAgent => {
                match self.switch_state_mapping.get_mut(&(agent, task)) {
                    None => {
                        let mut hmap_new: FastHM<SwitchTypeCache, Vec<SwitchMappings>> = FastHM::new();
                        hmap_new.insert(SwitchTypeCache::NextAgent, vec![SwitchMappings {
                            dynamic_link_idx: current_idx,
                            true_idx: next_idx
                        }]);
                        self.switch_state_mapping.insert((agent, task), hmap_new);
                    }
                    Some(x) => {
                        match x.get_mut(&SwitchTypeCache::NextAgent) {
                            None => {
                                x.insert(SwitchTypeCache::NextAgent, vec![SwitchMappings{
                                    dynamic_link_idx: current_idx,
                                    true_idx: next_idx
                                }]);
                            }
                            Some(v) => {
                                v.push(SwitchMappings {
                                    dynamic_link_idx: current_idx,
                                    true_idx: next_idx
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    fn incremental_modify_forward(
        &mut self,
        mdp: &mut MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>,
        next_idx: usize
    ) {
        let i = mdp.get_agent().unwrap();
        let j = mdp.get_task().unwrap();

        if j < (self.num_tasks - 1) as i32 {
            // The rewards hashmap will also require editing to emulate the process above
            //(SwitchMappings::new(idx, next_idx));
            // generate the transitions to the switch state
            // P(s, q==acc || rej, b=-1, s'=0, q'=-1)
            //     Accepting states first
            let done = mdp.get_labels(MDPLabel::Done).to_vec();
            // construct a state (-sink_count, -2) in which q \in done
            for (sb, qb) in done {
                mdp.transitions.insert(((sb, qb), -1), vec![((0, -1), 1f64)]);
                mdp.rewards.insert(((sb, qb), -1), [0.0, 0.0]);
                match mdp.pred_transitions.get_mut(&(0, -1)) {
                    None => {
                        mdp.pred_transitions
                            .insert((0, -1), vec![Pred { action: -1, state: (sb, qb) }]);
                    }
                    Some(x) => {
                        x.push(Pred{ action: -1, state: (sb, qb) });
                    }
                }
            }
            // We need to know the state index of the coded state (0, -1) pushed to the state
            // vector. This is easy since it will always just be the last state
            //
            // It really depends on which algorithm is being used as to whether the modified
            // states are being included or not
            mdp.states.push((0, -1));
            mdp.set_switch_label((0, -1 ));
            let idx = mdp.states.len() - 1;
            mdp.state_mapping.insert((0, -1), idx);
            mdp.reverse_state_mapping.insert(idx, (0, -1));
            // Map the switch states i.e. (0, -1) is just a dynamic link to its real address in
            // the value vector V, so we need to know the index mapping to get the value of
            // (s_{(0,0)}, q_{j + 1})
            self.update_switch_mapping(SwitchTypeCache::NextTask, idx, next_idx, i, j);
        }
    }

    fn incremental_modify_adjacent(
        &mut self,
        mdp: &mut MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>,
        next_idx: usize
    ) {
        let i = mdp.get_agent().unwrap();
        let j = mdp.get_task().unwrap();

        if i < (self.num_agents - 1) as i32 {
            // Coding the switch to the next agent as (-1, 0), that is the agent has changed
            // but the task remains the same
            mdp.states.push((-1, 0));
            // Index of the new state added
            let idx = mdp.states.len() - 1;
            mdp.set_switch_label((-1, 0));
            mdp.state_mapping.insert((-1, 0), idx);
            mdp.reverse_state_mapping.insert(idx, (-1, 0));
            // Map the switch states i.e. (-1, 0) is just a dynamic link to its real address in
            // the value vector V, so we need to know the index mapping to get the value of
            // (s_{(i + 1, 0)}, q_{j})
            // Adding the transition from an initial state to the next initial state
            let curr_init_state = mdp.init_state;
            mdp.transitions.insert((curr_init_state, -1), vec![((-1, 0), 1f64)]);
            mdp.pred_transitions.insert(
                (-1, 0),
                vec![Pred{ action: -1, state: curr_init_state }]
            );
            mdp.rewards.insert((curr_init_state, -1), [0.0, 0.0]);
            self.update_switch_mapping(SwitchTypeCache::NextAgent, idx, next_idx, i, j);
        }
    }

    pub fn print_mat_from_input(
        &self,
        mdp: &MDP<(i32, i32), Vec<(i32, i32)>, Vec<((i32, i32), f64)>, [f64; 2]>,
        agent: usize,
        task: usize,
        nr: usize,
        nc: usize,
        m: &[f64]
    ) {
        println!("Matrix[{}x{}]: (Agent: {}, Task: {})", nr, nc, agent, task);
        for r in 0..nr+1 {
            for c in 0..nc + 1 {
                if r == 0 {
                    if c == 0 {
                        print!("{0:width$}", "",width=5);
                    } else {
                        let (s, q) = mdp
                            .reverse_state_mapping
                            .get(&(c - 1))
                            .unwrap();
                        print!("[{},{}]", s, q)
                    }
                } else {
                    if c == 0 {
                        let (s, q) = mdp
                            .reverse_state_mapping
                            .get(&(r - 1))
                            .unwrap();
                        print!("[{},{}]", s, q)
                    } else {
                        print!("{:width$}", m[(c-1) * nr + (r-1)], width=5)
                    }
                }
            }
            println!();
        }
    }

    pub fn incremental_get_rand_proper_policy(&self, a: i32, t: i32, pi: &mut [f64]) {
        let avail_acts = self.available_actions.get(&(a, t)).unwrap();
        let proper_actions = self.proper_policies.get(&(a, t)).unwrap();
        for s in 0..*self.state_dims.get(&(a, t)).unwrap() {
            match proper_actions.get(&s) {
                Some(actions) => {
                    if s != *self.init_states.get(&(a, t)).unwrap() &&
                        actions.iter().any(|x| *x == -1) { // i.e this is the task handover state, like suc or fai
                        pi[s] = 0.;
                    } else {
                        pi[s] = *actions.into_iter()
                            .choose(&mut thread_rng()).unwrap() as f64 + 1.0;
                    }
                }
                None => {
                    // randomly select any action
                    let act = avail_acts.get(&s).unwrap().choose(&mut thread_rng()).unwrap();
                    pi[s] = *act as f64 + 1.0;
                }
            }
        }
    }
}


impl LPSolver for SCPM { }