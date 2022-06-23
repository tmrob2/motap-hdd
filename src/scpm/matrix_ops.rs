#![allow(non_snake_case)]

use crate::scpm::definition::{XpS, XpSprPr, SCPM};
use crate::definition::MDP;
use crate::{COO};
use crate::utils::number_fmts::write_f64_to_file;

/// Matrix Operations supporting the MOTAP Algorithm inputs
pub trait MatrixOps {
    fn incremental_construct_spblas_and_rewards(
        mdp: MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>,
        act_start: i32,
        act_end: i32,
        num_agents: usize,
        num_tasks: usize
    );
}

impl MatrixOps for SCPM {
    fn incremental_construct_spblas_and_rewards(
        mdp: MDP<XpS, Vec<XpS>, Vec<XpSprPr>, [f64; 2]>,
        act_start: i32,
        act_end: i32,
        num_agents: usize,
        num_tasks: usize
    ) {
        let agent = mdp.get_agent().unwrap();
        let task = mdp.get_task().unwrap();
        let msize = mdp.states
            .iter()
            .filter(|(s, q)| *s != -1 && *q != -1)
            .count();
        let nsize = mdp.states.len();
        for action in act_start..act_end {
            // gather the rows
            let mut rewards: Vec<f64> = vec![-f32::MAX as f64; msize * (num_tasks + num_agents)];
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
                        // enabled.
                    }
                    Some(reward) => {
                        // Either there is an action but its value is -infinity or some real
                        // reward exists. If it is the case that a reward is found but that
                        // reward is -inf, then all other agent, task rewards for this matrix row
                        // should also be -inf. The above essentially means that the action is
                        // not enabled for this state
                        for a in 0..num_agents {
                            if reward[0] != -f32::MAX as f64 {
                                rewards[a * msize + *row_idx] = 0.0;
                            }
                        }
                        for t in 0..num_tasks {
                            if reward[0] != -f32::MAX as f64 {
                                rewards[(num_agents + t) * msize + *row_idx] = 0.0;
                            }
                        }
                        rewards[agent as usize * msize + *row_idx] = reward[0];
                        rewards[(num_agents + task as usize) * msize + *row_idx] = reward[1];
                    }
                }
            }

            // construct a new sparse matrix to save
            let nnz = vals.len();
            let S = COO {
                nzmax: nnz as i32,
                nr: msize as i32,
                nc: nsize as i32,
                i: r,
                j: c,
                x: vals,
                nz: nnz as i32
            };
            let filename = format!("mat_{}_{}_{}.yml", agent, task, action);
            S.store_matrix_as_yaml(filename.as_str());
            // write the rewards vector to disk as well
            // check if the rewards directory exists
            let rewards_fname = format!("r_{}_{}_{}.txt", agent, task, action);
            let pth = format!("{}/rewards", std::env::var("SCPM_HOME").unwrap());
            write_f64_to_file(pth.as_str(), rewards_fname.as_str(), &rewards[..]);
        }
    }
}

