use std::f64::consts::PI;

use crate::fk::{L2, L3, L4, L5, L6, L7, L8, L9};
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};
use rayon::prelude::*;

use crate::fk::ee_pos;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Sqrt,
    Atan,
}

impl Distribution<Op> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Op {
        num_to_op(rng.gen_range(0..8))
    }
}

pub(crate) fn op_to_num(op: Op) -> usize {
    match op {
        Op::Add => 0,
        Op::Sub => 1,
        Op::Mul => 2,
        Op::Div => 3,
        Op::Sin => 4,
        Op::Cos => 5,
        Op::Sqrt => 6,
        Op::Atan => 7,
    }
}

pub(crate) fn num_to_op(n: usize) -> Op {
    match n {
        0 => Op::Add,
        1 => Op::Sub,
        2 => Op::Mul,
        3 => Op::Div,
        4 => Op::Sin,
        5 => Op::Cos,
        6 => Op::Sqrt,
        7 => Op::Atan,
        _ => Op::Atan,
    }
}

pub(crate) static CONST_REGS: usize = 13;
pub(crate) static VAR_REGS: usize = 10;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Chromosome {
    pub(crate) operator: Op,
    pub(crate) dest: usize,
    pub(crate) src1: usize,
    pub(crate) src2: usize,
}

fn safe_div(x: f64, y: f64) -> f64 {
    let divisor: f64 = if y == 0.0 { 0.0000001 } else { y };
    x / divisor
}

pub(crate) fn execute(regs: &mut [f64], genome: &[Chromosome]) {
    for chromo in genome {
        regs[chromo.dest] = match chromo.operator {
            Op::Add => regs[chromo.src1] + regs[chromo.src2],
            Op::Sub => regs[chromo.src1] - regs[chromo.src2],
            Op::Mul => regs[chromo.src1] * regs[chromo.src2],
            Op::Div => safe_div(regs[chromo.src1], regs[chromo.src2]),
            Op::Sin => regs[chromo.src1].sin(),
            Op::Cos => regs[chromo.src1].cos(),
            Op::Sqrt => regs[chromo.src1].abs().sqrt(),
            Op::Atan => regs[chromo.src1].atan2(regs[chromo.src2]),
        }
    }
}

pub(crate) fn init_registers(x: f64, y: f64, z: f64) -> Vec<f64> {
    let mut output = Vec::with_capacity(CONST_REGS + VAR_REGS);
    output.push(1.0);
    output.push(PI);
    output.push(L2);
    output.push(L3);
    output.push(L4);
    output.push(L5);
    output.push(L6);
    output.push(L7);
    output.push(L8);
    output.push(L9);
    output.push(x);
    output.push(y);
    output.push(z);
    output.resize(CONST_REGS + VAR_REGS, 0.0);
    output
}

pub(crate) fn init_population(
    n: usize,
    min_chromo: usize,
    max_chromo: usize,
) -> Vec<Vec<Chromosome>> {
    let mut population = Vec::with_capacity(n);

    for _ in 0..n {
        let chromos = thread_rng().gen_range(min_chromo..max_chromo);
        let mut individual = Vec::with_capacity(chromos);
        for _ in 0..chromos {
            individual.push(Chromosome {
                operator: thread_rng().gen(),
                dest: thread_rng().gen_range(CONST_REGS..(CONST_REGS + VAR_REGS)),
                src1: thread_rng().gen_range(0..(CONST_REGS + VAR_REGS)),
                src2: thread_rng().gen_range(0..(CONST_REGS + VAR_REGS)),
            });
        }
        population.push(individual);
    }

    population
}

pub(crate) fn evaluate_population_par(
    population: &Vec<Vec<Chromosome>>,
    test_angles: &[(f64, f64, f64)],
) -> Vec<f64> {
    let fitness = population.par_iter().map(|individual| {
        let mut error = 0.0;
        for (th_1, th_2, th_3) in test_angles {
            let (x, y, z) = ee_pos(*th_1, *th_2, *th_3);

            let mut regs = init_registers(x, y, z);
            execute(&mut regs, individual);

            let (x_p, y_p, z_p) =
                ee_pos(regs[CONST_REGS], regs[CONST_REGS + 1], regs[CONST_REGS + 2]);
            error += ((x - x_p).powi(2) + (y - y_p).powi(2) + (z - z_p).powi(2)).sqrt();
        }
        -error
    });

    fitness.collect()
}

pub(crate) fn tournament_select(
    sorted_fitness_with_i: &Vec<(f64, usize)>,
    t_size: usize,
    p_tour: f64,
) -> usize {
    let mut i = 0;
    while thread_rng().gen::<f64>() < p_tour && i < t_size && i < sorted_fitness_with_i.len() {
        i += 1;
    }
    sorted_fitness_with_i[i].1
}

pub(crate) fn binary_search(
    sorted_fitness_with_i: &Vec<(f64, usize)>,
    target_f: f64,
) -> &(f64, usize) {
    let mut lo = 0;
    let mut hi = sorted_fitness_with_i.len() - 1;

    let mut m: usize;
    while lo <= hi && hi > 1 {
        m = (hi + lo) / 2;
        if sorted_fitness_with_i[m].0 > target_f {
            lo = m + 1;
        } else if sorted_fitness_with_i[m].0 < target_f {
            hi = m - 1;
        }
    }

    if lo == 0 {
        if (sorted_fitness_with_i[0].0 - target_f).abs()
            < (sorted_fitness_with_i[1].0 - target_f).abs()
        {
            return &sorted_fitness_with_i[0];
        } else {
            return &sorted_fitness_with_i[1];
        }
    }
    if hi == sorted_fitness_with_i.len() - 1 {
        if (sorted_fitness_with_i[hi].0 - target_f).abs()
            < (sorted_fitness_with_i[hi - 1].0 - target_f).abs()
        {
            return &sorted_fitness_with_i[hi];
        } else {
            return &sorted_fitness_with_i[hi - 1];
        }
    }

    if (sorted_fitness_with_i[lo].0 - target_f).abs() < (sorted_fitness_with_i[hi].0 - target_f) {
        &sorted_fitness_with_i[lo]
    } else {
        &sorted_fitness_with_i[hi]
    }
}

// Fitness Uniform Optimization
// https://arxiv.org/abs/cs/0610126
pub(crate) fn fitness_uniform_select(sorted_fitness_with_i: &Vec<(f64, usize)>) -> usize {
    //let min_fit = sorted_fitness_with_i.first().unwrap().0 - 5.0;
    //let max_fit = sorted_fitness_with_i.last().unwrap().0 + 5.0;
    let min_fit = -50.0;
    let max_fit = 0.0;

    let f: f64 = thread_rng().gen::<f64>() * (max_fit - min_fit) + min_fit;

    binary_search(sorted_fitness_with_i, f).1
}

pub(crate) fn create_offspring(
    parent1: &Vec<Chromosome>,
    parent2: &Vec<Chromosome>,
    p_cross: f64,
    p_mut: f64,
    max_len: usize,
) -> (Vec<Chromosome>, Vec<Chromosome>) {
    let mut child1 = (*parent1).clone();
    let mut child2 = (*parent2).clone();

    let mut rng = thread_rng();

    if rng.gen::<f64>() < p_cross {
        let c_1_1 = rng.gen_range(0..child1.len() - 1);
        let c_1_2 = rng.gen_range(c_1_1 + 1..child1.len());

        let c_2_1 = rng.gen_range(0..child2.len() - 1);
        let c_2_2 = rng.gen_range(c_2_1 + 1..child2.len());

        let tmp1 = [&child1[0..c_1_1], &child2[c_2_1..c_2_2], &child1[c_1_2..]].concat();
        let tmp2 = [&child2[0..c_2_1], &child1[c_1_1..c_1_2], &child2[c_2_2..]].concat();

        child1 = tmp1;
        child2 = tmp2;

        child1.truncate(max_len);
        child2.truncate(max_len);
    }

    for chromo in &mut child1 {
        if rng.gen::<f64>() < p_mut {
            match rng.gen_range(0..3) {
                0 => {
                    chromo.operator = rng.gen();
                }
                1 => {
                    rng.gen_range(CONST_REGS..(CONST_REGS + VAR_REGS));
                }
                2 => {
                    rng.gen_range(0..(CONST_REGS + VAR_REGS));
                }
                _ => {
                    rng.gen_range(0..(CONST_REGS + VAR_REGS));
                }
            }
        }
    }

    (child1, child2)
}
