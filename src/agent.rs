use std::fmt::Debug;
use std::fs::OpenOptions;
use std::ops::Range;
use hashbrown::{HashMap};
use crate::definition::{MDP, Pred};
use crate::dfa::definition::{DFA2, Data};
use itertools::{Itertools, iproduct};
use crate::{fast_reverse_key_value_pairs, reverse_key_value_pairs};
use std::hash::Hash;
use std::collections::HashMap as SlowHashMap;
use std::io::BufReader;
use std::iter::FromIterator;
use serde::{Serialize, Deserialize};
use serde_json;

pub type ProductState = (i32, i32);
pub type ProductStateSpace = Vec<ProductState>;
pub type ProdPrimeTuple = (ProductState, f64);

#[derive(Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct MDPState {
    pub s: i32,
    pub q: i32
}

pub fn serialise_state_mapping(state_map: &SlowHashMap<usize, MDPState>, agent: i32, task: i32) {
    let pth = format!("{}/mappings", std::env::var("SCPM_HOME").unwrap());
    let filename = format!("{}/mdp_{}_{}.txt", pth, agent, task);
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename).unwrap();
    serde_json::to_writer_pretty(
        &file,
        state_map
    ).unwrap();
}

pub fn read_serialised_state_mapping(agent: i32, task: i32) -> Result<SlowHashMap<usize, MDPState>, Box<dyn std::error::Error>> {
    let pth = format!("{}/mappings", std::env::var("SCPM_HOME").unwrap());
    let filename = format!("{}/mdp_{}_{}.txt", pth, agent, task);
    let file = OpenOptions::new()
        .read(true)
        .open(filename).unwrap();
    let reader = BufReader::new(file);
    let u: SlowHashMap<usize, MDPState> = serde_json::from_reader(reader)?;
    Ok(u)
}

pub fn make_serialised_state_map(map: &HashMap<usize, ProductState>) -> SlowHashMap<usize, MDPState> {
    let mut rtn_map: SlowHashMap<usize, MDPState> = SlowHashMap::new();
    for (k, v) in map.iter() {
        rtn_map.insert(*k, MDPState{ s: v.0, q: v.1 });
    }
    rtn_map
}

#[derive(Clone)]
pub struct Robot<S: Hash + Eq, W> {
    pub states: Vec<S>,
    pub init_state: S,
    pub actions: std::ops::Range<i32>,
    pub labels: HashMap<S, W>,
    pub transitions: HashMap<(i32, i32), Vec<(i32, f64, W)>>, // reference to the state number not the state itself
    pub rewards: HashMap<(i32, i32), f64>,
    pub alphabet: Vec<W>,
    pub state_mapping: HashMap<S, usize>,
    pub reverse_state_mapping: HashMap<usize, S>
}

pub trait Agent<S, W: Clone> {
    fn action_space(&self) -> &std::ops::Range<i32>;

    fn num_actions(&self) -> usize;

    fn get_alphabet(&self) -> &[W];

    fn get_state_mapping(&self) -> &HashMap<S, usize>;

    fn get_reverse_state_mapping(&self) -> &HashMap<usize, S>;

    fn get_init_state(&self) -> &S;

    fn get_states(&self) -> &[S];

    fn get_transitions(&self) -> &HashMap<(i32, i32), Vec<(i32, f64, W)>>;

    fn get_rewards(&self) -> &HashMap<(i32, i32), f64>;

    fn set_state(&mut self, state: &S);

    fn insert_state_mapping(&mut self, k: &S, v: usize);

    fn insert_transition(&mut self, state: i32, action: i32, sprimes: Vec<(i32, f64, W)>);

    fn insert_reward(&mut self, state: i32, action: i32, reward: f64);

    fn set_reverse_state_mapping(&mut self);

    fn insert_word(&mut self, w: W);
}

pub trait MDPOps<S, F, W: Clone, T> where for<'a> F: Fn(&'a Data<W, T>) -> i32 {
    fn product(&self, task: &mut DFA2::<F, W, T>, agent: i32, task: i32, info: Option<T>)
               -> MDP<ProductState, ProductStateSpace, Vec<ProdPrimeTuple>, [f64; 2]>
        where Self: Agent<S, W>;

    fn label(&self, task: &DFA2::<F, W, T>)
             -> (ProductStateSpace, ProductStateSpace, ProductStateSpace, ProductStateSpace)
        where Self: Agent<S, W>;

    fn dfs(&self, task: &DFA2::<F, W, T>, info: Option<T>) -> Vec<bool> where Self: Agent<S, W>;

    fn rewards_fn(&self, state: &ProductState, dfa: &DFA2::<F, W, T>, r: Option<&f64>) -> f64;

    fn task_rewards_fn(&self, state: &ProductState, dfa: &DFA2::<F, W, T>, flag: Option<&f64>) -> f64;
}

impl<S, F, W, T: Copy> MDPOps<S, F, W, T> for Robot<S, W>
    where S: Hash + Eq + Debug, W: Clone + Default, for<'a> F: Fn(&'a Data<W, T>) -> i32 {
    /// Should only be the product set of states
    fn product(&self, task: &mut DFA2::<F, W, T>, agent: i32, t: i32, info: Option<T>)
               -> MDP<ProductState, ProductStateSpace, Vec<ProdPrimeTuple>, [f64; 2]>
        where Self: Agent<S, W> {
        let mut mdp: MDP<ProductState, ProductStateSpace, Vec<ProdPrimeTuple>, [f64; 2]> =
            MDP::make(self.num_actions() as i32);
        let mut states: ProductStateSpace = Vec::new();
        let mut state_mapping: HashMap<ProductState, usize> = HashMap::new();
        let mut product_transitions: HashMap<(ProductState, i32), Vec<ProdPrimeTuple>> = HashMap::new();
        let mut predecessor_transitions: HashMap<ProductState, Vec<Pred>> = HashMap::new();
        let mut product_rewards: HashMap<(ProductState, i32), [f64; 2]> = HashMap::new();
        let mut state_counter: usize = 0;
        task.sink(self.get_alphabet().to_vec(), info);
        // Use DFS to determine the reachable set of states
        let visited = self.dfs(task, info);
        for (ix, (s, q)) in self.get_states().iter()
            .cartesian_product(task.get_states().iter()).enumerate() {
            if visited[ix] {
                // continue with the product calculation
                let sidx = *self.get_state_mapping().get(s).unwrap() as i32;
                states.push((sidx as i32 , *q));
                state_mapping.insert((sidx as i32, *q), state_counter);
                state_counter += 1;
                for a in self.action_space().start..self.action_space().end {
                    let mut sprimes: Vec<(ProductState, f64)> = Vec::new();
                    match self.get_transitions().get(&(sidx , a)) {
                        None => {}
                        Some(v) => {
                            for (sprime, p, w) in v.iter() {
                                let qprime = task.call(&Data{ q: *q, w: w.clone(), info: info });
                                sprimes.push(((*sprime, qprime), *p));
                                match predecessor_transitions.get_mut(&(*sprime, qprime)) {
                                    None => {
                                        // add the key value to the predecessor
                                        predecessor_transitions.insert(
                                            (*sprime, qprime), vec![Pred{ action: a, state: (sidx, *q) }]
                                        );
                                    }
                                    Some(pred) => {
                                        pred.push(Pred{ action: a, state: (sidx, *q) });
                                    }
                                }
                            }
                        }
                    }
                    let r = self.get_rewards().get(&(sidx, a));
                    let xp_rewards = self.rewards_fn(&(sidx, *q), task, r);
                    let task_rewards = self.task_rewards_fn(&(sidx, *q), task, r);
                    product_rewards.insert(((sidx, *q), a), [xp_rewards, task_rewards]);
                    if !sprimes.is_empty() {
                        product_transitions.insert(((sidx, *q), a), sprimes);
                    }
                }
            }
        }

        let reverse_mapping = fast_reverse_key_value_pairs(&state_mapping);
        let init = *self.get_state_mapping().get(&self.get_init_state()).unwrap();
        mdp.init_state = (init as i32, task.get_init());
        mdp.states = states;
        mdp.state_mapping = state_mapping;
        mdp.reverse_state_mapping = reverse_mapping;
        mdp.transitions = product_transitions;
        mdp.pred_transitions = predecessor_transitions;
        mdp.rewards = product_rewards;
        mdp.set_agent(agent);
        mdp.set_task(t);
        let (suc, fai, done, ec) =
            self.label(task);
        mdp.set_labels(suc, fai, done, ec);
        mdp
    }

    fn label(&self, task: &DFA2::<F, W, T>)
             -> (ProductStateSpace, ProductStateSpace, ProductStateSpace, ProductStateSpace)
        where for<'a> F: Fn(&'a Data<W, T>) -> i32, Self: Agent<S, W> {
        let mut succ: Vec<ProductState> = Vec::new();
        let mut fai: Vec<ProductState> = Vec::new();
        let mut done: Vec<ProductState> = Vec::new();
        let mut ec: Vec<ProductState> = Vec::new();

        // Label successful states
        let init_state = *self.get_state_mapping().get(&self.init_state).unwrap() as i32;
        for (s, q) in iproduct!(vec![init_state].iter(), task.get_accepting().iter()) {
            succ.push((*s, *q));
        }
        // Label task failures
        for (s, q) in iproduct!(vec![init_state].iter(), task.get_rejecting().iter()) {
            fai.push((*s, *q));
        }

        // label regenerating
        for (s, q) in iproduct!(vec![init_state].iter(), task.get_done().iter()) {
            done.push((*s, *q));
        }

        // Labelling the end components
        let a = task.get_done().iter().map(|x| *x).collect::<Vec<i32>>();
        //a.extend_from_slice(&task.accepting[..]);
        for (s, q) in iproduct!(self.get_states().iter(), a.iter()) {
            let si32 = *self.get_state_mapping().get(s).unwrap() as i32;
            ec.push((si32, *q));
        }

        (succ, fai, done, ec)
    }

    fn dfs(&self, task: &DFA2::<F, W, T>, info: Option<T>) -> Vec<bool>
        where for<'a> F: Fn(&'a Data<W, T>) -> i32, Self: Agent<S, W> {
        let mut stack: Vec<(i32, i32)> = Vec::new();
        // what is the state space, this should probably be input by the
        // states are going to be of type (S, i32)
        let state_space: Vec<((i32, i32), usize)> = self.get_states()
            .iter()
            .cartesian_product(task.get_states().iter())
            .enumerate()
            .map(|(ix, (x1, x2))|
                ((*self.get_state_mapping().get(x1).unwrap() as i32, *x2), ix)
            )
            .collect();
        let state_mapping: HashMap<(i32, i32), usize> = HashMap::from_iter(state_space.into_iter());
        let state_space_len = self.get_states().len() * task.get_states().len();
        let mut visited: Vec<bool> = vec![false; state_space_len];
        let agent_init = &self.get_init_state();
        let agent_init_idx = *self.get_state_mapping().get(agent_init).unwrap() as i32;
        let task_init_idx = task.get_init();
        stack.push((agent_init_idx, task_init_idx));
        let init_idx = state_mapping.get(&(agent_init_idx, task_init_idx)).unwrap();
        visited[*init_idx] = true;
        while !stack.is_empty() {
            let (s, q) = stack.pop().unwrap();
            for a in self.action_space().start..self.action_space().end {
                match self.get_transitions().get(&(s, a)) {
                    None => {}
                    Some(v) => {
                        for (sprime, _, w) in v.iter() {
                            let qprime: i32 = task.call(&Data{q, w: w.clone(), info });
                            let next_state: (i32, i32) = (*sprime, qprime);
                            //println!("sprime: {:?}, q: {}", *sprime, q);
                            let sprime_idx = match state_mapping.get(&next_state) {
                                Some(x) => { x }
                                None => { panic!(
                                    "this error usually occurs when the DFA accepting/rejecting \
                                    positions are not correct or the correct number of DFA states \
                                    has not been defined, {:?}",
                                    next_state
                                )}
                            };
                            if !visited[*sprime_idx] {
                                stack.push(next_state);
                                visited[*sprime_idx] = true;
                            }
                        }
                    }
                }
            }
        }
        visited
    }
    // might get rid of assigning 0s in the rewards function here and then assign zero to the last
    // task and the last agent
    fn rewards_fn(&self, state: &ProductState, dfa: &DFA2::<F, W, T>, r: Option<&f64>) -> f64 {
        match state {
            //(_, x) if dfa.get_done().iter().any(|z| z == x)
            //    || dfa.get_accepting().iter().any(|z| z == x) => {
            (_, x) if dfa.get_accepting().iter().any(|z| z == x) => {
                match r {
                    Some(_) => { 0.0 }
                    None => { -f32::MAX as f64 }
                }
            }
            _ =>  {
                match r {
                    Some(val) => { -*val },
                    None => { -f32::MAX as f64 }
                }
            }
        }
    }

    fn task_rewards_fn(&self, state: &ProductState, dfa: &DFA2::<F, W, T>, flag: Option<&f64>) -> f64 {
        match flag {
            Some(_) => {
                match state {
                    (_, x) if dfa.get_accepting().iter().any(|z| z == x) => { 1.0 }
                    _ => { 0.0 }
                }
            }
            None => { -f32::MAX as f64 }
        }
    }
}

impl<S, W> Agent<S, W> for Robot<S, W>
where S: Eq + Hash + Copy, W: Clone {
    fn action_space(&self) -> &Range<i32> {
        &self.actions
    }

    fn num_actions(&self) -> usize {
        self.actions.end as usize
    }

    fn get_alphabet(&self) -> &[W] {
        &self.alphabet[..]
    }

    fn get_state_mapping(&self) -> &HashMap<S, usize> {
        &self.state_mapping
    }

    fn get_reverse_state_mapping(&self) -> &HashMap<usize, S> {
        &self.reverse_state_mapping
    }

    fn get_init_state(&self) -> &S {
        &self.init_state
    }

    fn get_states(&self) -> &[S] {
        &self.states[..]
    }

    fn get_transitions(&self) -> &HashMap<(i32, i32), Vec<(i32, f64, W)>> {
        &self.transitions
    }

    fn get_rewards(&self) -> &HashMap<(i32, i32), f64> {
        &self.rewards
    }

    fn set_state(&mut self, state: &S) {
        self.states.push(*state)
    }

    fn insert_state_mapping(&mut self, k: &S, v: usize) {
        self.state_mapping.insert(*k, v);
    }

    fn insert_transition(&mut self, state: i32, action: i32, sprimes: Vec<(i32, f64, W)>) {
        self.transitions.insert((state, action), sprimes);
    }

    fn insert_reward(&mut self, state: i32, action: i32, reward: f64) {
        self.rewards.insert((state, action), reward);
    }

    fn set_reverse_state_mapping(&mut self) {
        self.reverse_state_mapping = reverse_key_value_pairs(&self.state_mapping);
    }

    fn insert_word(&mut self, w: W) {
        self.alphabet.push(w);
    }
}