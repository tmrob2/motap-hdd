use hashbrown::HashMap;
use gurobi::{attr, Continuous, LinExpr, Model, param, Status, Greater, Equal, Less, Var};

pub trait LPSolver {
    /// The linear programming solver specific to a MOTAP problem
    ///
    /// We are interested in generating weight vectors wbar representing separating hyperplanes.
    /// To do this we implement a linear programming heuristic. Given some point r (derived as
    /// the y0 values for each objective k) and a set of availability points X, choose wbar to
    /// maximise min_{x in X}(wq - wx).
    ///
    /// In practice this can be done by constructing the linear programming problem:
    ///
    /// constraint 1: Sum w_i = 1
    /// constraint 2..{X}+1 = w_i . (q_i - x_i ) >= d
    ///
    /// Even for many tasks and many agents, these problems are generally very small and
    /// will be quickly solved
    ///
    /// Note on the ordering of variables:
    ///
    fn gurobi_solver(&self, h: &HashMap<usize, Vec<f64>>, t: &[f64], dim: &usize) -> Option<Vec<f64>> {
        let mut env = gurobi::Env::new("").unwrap();
        env.set(param::OutputFlag, 0).ok();
        env.set(param::LogToConsole, 0).ok();
        env.set(param::InfUnbdInfo, 1).ok();
        //env.set(param::FeasibilityTol,10e-9).unwrap();
        env.set(param::NumericFocus,2).ok();
        let mut model = Model::new("model1", &env).unwrap();

        // add variables
        let mut vars: HashMap<String, gurobi::Var> = HashMap::new();
        for i in 0..*dim {
            vars.insert(format!("w{}", i), model.add_var(
                &*format!("w{}", i),
                Continuous,
                0.0,
                0.0,
                1.0,
                &[],
                &[]).unwrap()
            );
        }
        let d = model.add_var(
            "d", Continuous, 0.0, -gurobi::INFINITY, gurobi::INFINITY, &[], &[]
        ).unwrap();

        model.update().unwrap();
        let mut w_vars = Vec::new();
        for i in 0..*dim {
            let w = vars.get(&format!("w{}", i)).unwrap();
            w_vars.push(w.clone());
        }
        let t_expr = LinExpr::new();
        let t_expr1 = t_expr.add_terms(&t[..], &w_vars[..]);
        let t_expr2 = t_expr1.add_term(1.0, d.clone());
        let t_expr3 = t_expr2.add_constant(-1.0);
        model.add_constr("t0", t_expr3, gurobi::Greater, 0.0).ok();

        for ii in 0..h.len() {
            let expr = LinExpr::new();
            let expr1 = expr.add_terms(&h.get(&ii).unwrap()[..], &w_vars[..]);
            let expr2 = expr1.add_term(1.0, d.clone());
            let expr3 = expr2.add_constant(-1.0);
            model.add_constr(&*format!("c{}", ii), expr3, gurobi::Less, 0.0).ok();
        }
        let w_expr = LinExpr::new();
        let coeffs: Vec<f64> = vec![1.0; *dim];
        let final_expr = w_expr.add_terms(&coeffs[..], &w_vars[..]);
        model.add_constr("w_final", final_expr, gurobi::Equal, 1.0).ok();

        model.update().unwrap();
        model.set_objective(&d, gurobi::Maximize).unwrap();
        model.optimize().unwrap();
        let mut varsnew = Vec::new();
        for i in 0..*dim {
            let var = vars.get(&format!("w{}", i)).unwrap();
            varsnew.push(var.clone());
        }
        let val = model.get_values(attr::X, &varsnew[..]).unwrap();
        println!("model: {:?}", model.status());
        if model.status().unwrap() == Status::Infeasible {
            None
        } else {
            Some(val)
        }
    }

    /// Determines a weighting for each scheduler to follow at the start of each task
    ///
    /// costs => (i: agent, j: task, k: hull point)
    /// probs => (j: task, k: hull point)
    fn gurobi_task_witness(
        &self,
        costs: &HashMap<(i32, i32, i32), f64>,
        probs: &HashMap<(i32, i32), f64>,
        target: &[f64],
        num_sols: usize,
        num_tasks: usize,
        num_agents: usize
    ) -> Result<HashMap<i32, Vec<f64>>, Box<dyn std::error::Error>> {
        let mut env = gurobi::Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        env.set(param::LogToConsole, 0).unwrap();
        let mut model = Model::new("witness model", &env).unwrap();
        let mut solution: HashMap<i32, Vec<f64>> = HashMap::new();

        let mut W: HashMap<String, gurobi::Var> = HashMap::new();
        //println!("|hullset|: {}", hullset.len());
        for k in 0..num_sols {
            for j in 0..num_tasks {
                W.insert(
                    format!("w_{}_{}", j, k),
                    model
                        .add_var(
                            &*format!("w_{}_{}", j, k),
                            Continuous,
                            0.0,
                            0.00,
                            1.0,
                            &[],
                            &[]
                        ).unwrap());
            }
        }

        let dummy = model
            .add_var(
                "dummy",
                Continuous,
                0.0,
                0.0,
                0.0,
                &[],
                &[])
            .unwrap();
        model.update().unwrap();

        // Construct the cost equations:
        //  For each agent, we want a linear equation which determines the weight for each scheduler
        //  for some task j => w_j_k, this is somewhat different to the paper definition which is just
        //  w_j
        for i in 0..num_agents {
            let mut l: LinExpr = LinExpr::new();
            for j in 0..num_tasks {
                for k in 0..num_sols {
                    match costs.get(&(i as i32, j as i32, k as i32)) {
                        None => { }
                        Some(c) => {
                            let w = W.get(&*format!("w_{}_{}", j, k)).unwrap();
                            l = l.clone().add_term( *c, w.clone());
                        }
                    }
                }
            }
            model.add_constr(
                &*format!("c_{}", i),
                l,
                Greater,
                target[i]
            )?;
        }

        model.update()?;

        // Construct the task probability constraints
        //  For each task, the probability of completing the task must be above some target t
        for j in 0..num_tasks {
            let mut l: LinExpr = LinExpr::new();
            for k in 0..num_sols {
                let p = probs.get(&(j as i32, k as i32)).unwrap();
                let w = W.get(&*format!("w_{}_{}", j, k)).unwrap();
                l = l.clone().add_term(*p, w.clone());
            }
            model.add_constr(
                &*format!("c_{}", num_tasks + j),
                l,
                Greater,
                target[num_agents + j]
            )?;
        }

        model.update()?;

        // for each task the weighted sum must add to 1
        for j in 0..num_tasks {
            let mut l: LinExpr = LinExpr::new();
            let coeffs: Vec<f64> = vec![1.0; num_sols];
            let mut wk: Vec<_> = Vec::new();
            for k in 0..num_sols {
                wk.push(W.get(&*format!("w_{}_{}", j, k)).unwrap().clone());
            }
            l = l.add_terms(&coeffs[..], &wk[..]);
            model.add_constr(&*format!("c_weights_{}", j), l, Equal, 1.0)?;
        }

        model.update()?;

        model.set_objective(dummy, gurobi::Maximize)?;

        model.optimize()?;

        // construct a set of vectors for eack task, this will be the weighting
        // for each scheduler for each task

        for j in 0..num_tasks {
            let mut vars = Vec::new();
            for k in 0..num_sols {
                let var = W.get(&format!("w_{}_{}",j, k)).unwrap();
                vars.push(var.clone());
            }
            match model.get_values(attr::X, &vars[..]) {
                Ok(x) => { solution.insert(j as i32, x); }
                Err(_) => { panic!("Unable to retrieve var") }
            }
        }

        Ok(solution)
    }

    fn gurobi_witness(
        &self,
        hullset: &HashMap<usize, Vec<f64>>,
        target: &[f64]) -> Option<Vec<f64>> {
        //t env = Env::new().unwrap();
        let mut env = gurobi::Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        env.set(param::LogToConsole, 0).unwrap();
        let mut model = Model::new("model2", &env).unwrap();


        let mut v: HashMap<String, gurobi::Var> = HashMap::new();
        //println!("|hullset|: {}", hullset.len());
        for i in 0..hullset.len() {
            v.insert(
                format!("u{}", i),
                model
                    .add_var(
                    &*format!("u{}", i),
                    Continuous,
                    0.0,
                    0.00,
                    1.0,
                    &[],
                    &[]
                    ).unwrap());
        }
        let dummy = model
            .add_var(
                "dummy",
                Continuous,
                0.0,
                0.0,
                0.0,
                &[],
                &[])
            .unwrap();
        model.update().unwrap();

        let mut u_vars  = Vec::new();
        for i in 0..v.len(){
            let u = v.get(&format!("u{}", i)).unwrap();
            u_vars.push(u.clone());
        }

        let mut q_transpose: Vec<Vec<f64>> = Vec::new();
        for i in 0..target.len() {
            let mut q = Vec::new();
            for j in 0..hullset.len() {
                q.push(hullset.get(&j).unwrap()[i]);
            }
            q_transpose.push(q);
        }

        for i in 0..target.len() {
            let q_new = &q_transpose[i];
            let ui_expr = LinExpr::new();
            let ui_expr1 = ui_expr.add_terms(&q_new[..], &u_vars[..]);
            model.add_constr(
                &*format!("c{}", i),
                ui_expr1,
                Greater,
                target[i]).ok();
        }
        let u_expr = LinExpr::new();
        let coefs: Vec<f64> = vec![1.0; hullset.len()];
        let final_expr = u_expr.add_terms( &coefs[..], &u_vars[..]);
        model.add_constr("u_final", final_expr, gurobi::Equal, 1.0).ok();

        model.update().unwrap();
        model.set_objective(dummy,gurobi::Maximize).unwrap();

        model.optimize().unwrap();

        let mut vars = Vec::new();
        for i in 0..hullset.len() {
            let var = v.get(&format!("u{}",i)).unwrap();
            vars.push(var.clone());
        }
        match model.get_values(attr::X, &vars[..]) {
            Ok(x) => {Some(x)}
            Err(e) => { println!("unable to retrieve var because: {:?}", e); None}
        }
    }

    /// l is the cardinality of the hullset
    /// n is the number of objectives
    fn min_distance_to_target(
        hullset: &hashbrown::HashMap<usize, Vec<f64>>,
        target: &[f64],
        l: usize,
        n: usize
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let mut env = gurobi::Env::new("").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        env.set(param::LogToConsole, 0).unwrap();
        let mut model = Model::new("model2", &env).unwrap();


        let mut v: HashMap<String, gurobi::Var> = HashMap::new();
        for k in 0..l {
            v.insert(
                format!("lambda{}", k),
                model
                    .add_var(
                        &*format!("lambda{}", k),
                        Continuous,
                        0.0,
                        0.00,
                        1.0,
                        &[],
                        &[]
                    )?);
        }

        let epsilon = model.add_var(
            "epsilon", Continuous, 1.0, -gurobi::INFINITY, gurobi::INFINITY, &[], &[]
        )?;

        model.update()?;

        for j in 0..n {
            let mut e = LinExpr::new();
            // greater than epsilon constraint
            for k in 0..l {
                let x_kj = hullset.get(&k).unwrap()[j];
                let lambda = v.get(&*format!("lambda{}",k)).unwrap();
                e = e.clone().add_term(x_kj, lambda.clone());
            }
            e = e.clone().add_term(1.0, epsilon.clone());
            model.add_constr(
                &*format!("c_{}0", j),
                e,
                Greater,
                target[j]
            )?;
            let mut e = LinExpr::new();
            // greater than epsilon constraint
            for k in 0..l {
                let x_kj = hullset.get(&k).unwrap()[j];
                let lambda = v.get(&*format!("lambda{}",k)).unwrap();
                e = e.clone().add_term(x_kj, lambda.clone());
            }
            e = e.clone().add_term(-1.0, epsilon.clone());
            model.add_constr(
                &*format!("c_{}0", j),
                e,
                Less,
                target[j]
            )?;
        }
        model.update()?;

        let mut e = LinExpr::new();
        let coeffs: Vec<f64> = vec![1.0; l];
        let mut lambdas: Vec<_> = Vec::new();
        for k in 0..l {
            lambdas.push(v.get(&*format!("lambda{}",k)).unwrap().clone());
        }
        e = e.add_terms(&coeffs[..], &lambdas[..]);
        model.add_constr("lambda_sum", e, Equal, 1.0)?;

        model.update()?;
        model.set_objective(epsilon.clone(), gurobi::Minimize)?;
        model.optimize()?;
        let vars: Vec<Var> = vec![epsilon.clone()];
        let result = match model.get_values(attr::X, &vars[..]) {
            Ok(x) => { x[0] }
            Err(_) => { panic!(
                "Unable to retrieve variable from solution! \n \n
                Usually means that the LP could not find a solution for the given set of constraints."
            ) }
        };
        Ok(result)
    }
}

