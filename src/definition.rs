use std::hash::Hash;
use hashbrown::{HashMap as FastHM, HashSet};
use serde::Serialize;

pub enum MDPLabel {
    Suc,
    Fai,
    Done,
    EC,
    Sw
}

#[derive(Clone)]
/// MDP labelling is specifically for model checking
/// and therefore is useful in the scenarios where MDP Automata products are used to
/// check formal logic sequences on MDP structures.
struct LabellingAssociation<S> {
    success: Vec<S>,
    fail: Vec<S>,
    done: Vec<S>,
    ec: Vec<S>,
    sw: Vec<S>
}

#[derive(Copy, Clone, Debug)]
/// Struct representing the predecessor of a state and the action taken to get to the successor
/// state
pub struct Pred {
    pub action: i32,
    pub state: (i32, i32)
}

/// S - The state type
///
/// T: StateSpace type which is a convenience type because it will be Vec<S>
///
/// A: Action type
///
/// U: transition to type which could be a Vec<(state, prob)>, or could be (state, prob) tuple,
/// or state, it is up the the requirements of the MDP
///
/// R: Rewards type, in multiobjective problems this could be [f64; 2], but also could be scalar
#[derive(Clone)]
pub struct MDP<S, T, U, R> {
    pub states: T,
    pub actions: std::ops::Range<i32>,
    pub state_mapping: FastHM<S, usize>,
    pub reverse_state_mapping: FastHM<usize, S>,
    pub init_state: S,
    pub transitions: FastHM<(S, i32), U>,
    pub pred_transitions: FastHM<(i32, i32), Vec<Pred>>,
    pub rewards: FastHM<(S, i32), R>,
    label: Option<LabellingAssociation<S>>,
    agent: Option<i32>,
    task: Option<i32>,
    proper_policy: FastHM<(i32, i32), HashSet<i32>>
}

impl<S, T, U, R> MDP<S, T, U, R>
    where T: std::default::Default, S: Hash + Eq + Default, S: Serialize {
    /// na => number of actions
    pub fn make(na: i32) -> MDP<S, T, U, R> {
        MDP {
            states: Default::default(),
            actions: (0..na),
            state_mapping: FastHM::new(),
            reverse_state_mapping: FastHM::new(),
            init_state: Default::default(),
            transitions: FastHM::new(),
            pred_transitions: Default::default(),
            rewards: FastHM::new(),
            label: None,
            agent: None,
            task: None,
            proper_policy: Default::default()
        }
    }

    pub fn set_agent(&mut self, agent: i32) {
        self.agent = Some(agent);
    }

    pub fn set_task(&mut self, task: i32) {
        self.task = Some(task);
    }

    pub fn get_agent(&self) -> Option<i32> {
        self.agent
    }

    pub fn get_task(&self) -> Option<i32> {
        self.task
    }

    pub fn set_labels(&mut self, suc: Vec<S>, fai: Vec<S>, done: Vec<S>, ec: Vec<S>) {
        let label_assoc = LabellingAssociation {
            success: suc,
            fail: fai,
            done,
            ec,
            sw: vec![]
        };
        self.label = Some(label_assoc);
    }

    pub fn set_switch_label(&mut self, sw: S) {
        self.label.as_mut().unwrap().sw.push(sw);
    }

    pub fn get_labels(&self, label_type: MDPLabel) -> &[S] {
        // Make sure that we are calling a get function on something that exists
        assert!(self.label.is_some());
        match label_type {
            MDPLabel::Suc => { &self.label.as_ref().unwrap().success[..] }
            MDPLabel::Fai => { &self.label.as_ref().unwrap().fail[..] }
            MDPLabel::Done => { &self.label.as_ref().unwrap().done[..] }
            MDPLabel::EC => { &self.label.as_ref().unwrap().ec[..] }
            MDPLabel::Sw => { &self.label.as_ref().unwrap().sw[..] }
        }
    }

    pub fn set_proper_policy(&mut self, policy: FastHM<(i32, i32), HashSet<i32>>) {
        self.proper_policy = policy
    }

    pub fn get_proper_policy(&self) -> &FastHM<(i32, i32), HashSet<i32>> {
        &self.proper_policy
    }

    pub fn insert_propert_policy(&mut self, state: (i32, i32), action: i32) -> Result<(), &'static str> {
        match self.proper_policy.get_mut(&state) {
            Some(x) => {
                x.insert(action);
                return Ok(())
            }
            None => {
                return Err("state does not exist, cannot insert action")
            }
        };
    }
}

