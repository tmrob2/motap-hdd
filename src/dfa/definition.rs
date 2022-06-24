use std::iter::FromIterator;
use hashbrown::{HashSet as FastSet};

pub type Tfunc2 = dyn Fn(&i32, &str, i32) -> Option<i32>;

#[allow(non_snake_case)]
#[derive(Debug, PartialEq)]
pub struct Sink{
    pub ix: usize,
    pub Q: FastSet<i32>
}

impl Sink {
    pub fn new(ix: usize, val: i32) -> Sink {
        Sink {
            ix,
            Q: FastSet::from_iter(vec![val].into_iter())
        }
    }
}

#[derive(Clone)]
/// Deterministic Finite Automaton definition
///
/// N: number of states - size of DFA
///
/// extra_data<T> -> is a generic extra field, could be used for repeats
///
pub struct DFA {
    init: i32,
    pub states: Vec<i32>,
    pub accepting: Vec<i32>,
    pub rejecting: Vec<i32>,
    done: FastSet<i32>,
}

impl DFA {
    pub fn init(init: i32, states: &[i32], acc: &[i32], rej: Option<&[i32]>) -> DFA {
        let done: FastSet<i32> = FastSet::from_iter(rej.as_ref().unwrap().iter().cloned());
        DFA {
            init,
            states: states.to_vec(),
            accepting: acc.to_vec(),
            rejecting: if rej.is_some() { rej.as_ref().unwrap().to_vec() } else { vec![] },
            done,
        }
    }

    pub fn get_init(&self) -> i32 {
        self.init
    }

    pub fn get_done(&self) -> &FastSet<i32> {
        &self.done
    }

    pub fn set_done(&mut self, val: i32) {
        self.done.insert(val);
    }

    pub fn sink2<A, T>(&mut self, item: &A, words: &[T]) where A: DFANext<T>, T: Clone {
        let finished = self.accepting.to_vec();
        let rejecting = self.rejecting.to_vec();
        for q in rejecting {
            self.set_done(q);
        }
        for q in finished.iter() {
            for w in words.iter().cloned() {
                let qprime = item.goto(q, w);
                self.set_done(qprime.unwrap());
            }
        }
    }
}

pub trait DFANext<T> {
    fn goto(&self, q: &i32, w: T) -> Option<i32>;
}

pub struct Data<W, T> {
    pub q: i32,
    pub w: W,
    pub info: Option<T>
}

pub struct DFA2<F, W, T> {
    pub init: i32,
    pub states: Vec<i32>,
    pub accepting: Vec<i32>,
    pub rejecting: Vec<i32>,
    pub data: Data<W, T>,
    pub tfunc: F,
    pub done: FastSet<i32>
}

impl<F, W, T> DFA2<F, W, T> where F: Fn(&Data<W, T>) -> i32, W: Clone + Default, T: Clone {
    pub fn init(init: i32, states: &[i32], accepting: &[i32], rejecting: &[i32], tfunc: F, info: Option<T>) -> DFA2<F, W, T> {
        DFA2 {
            init,
            states: states.to_vec(),
            accepting: accepting.to_vec(),
            rejecting: rejecting.to_vec(),
            data: Data {q: 0, w: Default::default(), info},
            tfunc,
            done: Default::default()
        }
    }

    pub fn call(&self, data: &Data<W, T>) -> i32 {
        (self.tfunc)(data)
    }

    pub fn get_states(&self) -> &[i32] {
        &self.states[..]
    }

    pub fn get_accepting(&self) -> &[i32] {
        &self.accepting[..]
    }

    pub fn get_rejecting(&self) -> &[i32] {
        &self.rejecting[..]
    }

    pub fn get_init(&self) -> i32 {
        self.init
    }

    pub fn get_done(&self) -> &FastSet<i32> {
        &self.done
    }

    pub fn set_done(&mut self, val: i32) {
        self.done.insert(val);
    }

    pub fn sink(&mut self, alphabet: Vec<W>, info: Option<T>) {
        let finished = self.accepting.to_vec();
        let rejecting = self.rejecting.to_vec();
        for q in rejecting {
            self.set_done(q);
        }
        for q in finished.iter() {
            for w in alphabet.iter().cloned() {
                let input = match info.as_ref() {
                    Some(x) => Some(x.clone()),
                    None => None
                };
                let qprime: i32 = (self.tfunc)(&Data{q: *q, w, info: input});
                self.set_done(qprime);
            }
        }
    }
}

