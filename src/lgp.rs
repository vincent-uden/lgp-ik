use crate::fk::{L2, L3, L4, L5, L6, L7, L8, L9};
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};
use rayon::prelude::*;

use crate::fk::ee_pos;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Op {
    ADD,
    SUB,
    MUL,
    DIV,
    SIN,
    COS,
    SQT,
    ATAN,
}

impl Distribution<Op> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Op {
        return num_to_op(rng.gen_range(0..7));
    }
}

pub(crate) fn op_to_num(op: Op) -> usize {
    return match op {
        Op::ADD => 0,
        Op::SUB => 1,
        Op::MUL => 2,
        Op::DIV => 3,
        Op::SIN => 4,
        Op::COS => 5,
        Op::SQT => 6,
        Op::ATAN => 7,
    };
}

pub(crate) fn num_to_op(n: usize) -> Op {
    return match n {
        0 => Op::ADD,
        1 => Op::SUB,
        2 => Op::MUL,
        3 => Op::DIV,
        4 => Op::SIN,
        5 => Op::COS,
        6 => Op::SQT,
        7 => Op::ATAN,
        _ => Op::ADD,
    };
}

pub(crate) static CONST_REGS: usize = 10;
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

    return x / divisor;
}

pub(crate) fn execute(regs: &mut [f64], genome: &[Chromosome]) {
    for chromo in genome {
        regs[chromo.dest] = match chromo.operator {
            Op::ADD => regs[chromo.src1] + regs[chromo.src2],
            Op::SUB => regs[chromo.src1] - regs[chromo.src2],
            Op::MUL => regs[chromo.src1] * regs[chromo.src2],
            Op::DIV => safe_div(regs[chromo.src1], regs[chromo.src2]),
            Op::SIN => regs[chromo.src1].sin(),
            Op::COS => regs[chromo.src1].cos(),
            Op::SQT => regs[chromo.src1].abs().sqrt(),
            Op::ATAN => regs[chromo.src1].atan2(regs[chromo.src2]),
        }
    }
}

pub(crate) fn init_registers(th_1: f64, th_2: f64, th_3: f64) -> Vec<f64> {
    let mut output = Vec::with_capacity(CONST_REGS + VAR_REGS);
    output.push(1.0);
    output.push(0.5);
    output.push(L2);
    output.push(L3);
    output.push(L4);
    output.push(L5);
    output.push(L6);
    output.push(L7);
    output.push(L8);
    output.push(L9);
    output.push(th_1);
    output.push(th_2);
    output.push(th_3);
    for _ in 5..(CONST_REGS + VAR_REGS) {
        output.push(0.0);
    }
    return output;
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

    return population;
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

            let (x_p, y_p, z_p) = ee_pos(
                regs[CONST_REGS + 0],
                regs[CONST_REGS + 1],
                regs[CONST_REGS + 2],
            );
            error += ((x - x_p).powi(2) + (y - y_p).powi(2) + (z - z_p).powi(2)).sqrt();
        }
        return -error;
    });

    return fitness.collect();
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
    return sorted_fitness_with_i[i].1;
}

pub(crate) fn create_offspring(
    parent1: &Vec<Chromosome>,
    parent2: &Vec<Chromosome>,
    p_cross: f64,
    p_mut: f64,
) -> (Vec<Chromosome>, Vec<Chromosome>) {
    const MAX_LEN: usize = 100;
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

        child1.truncate(MAX_LEN);
        child2.truncate(MAX_LEN);
    }

    for i in 0..child1.len() {
        if rng.gen::<f64>() < p_mut {
            match rng.gen_range(0..3) {
                0 => {
                    child1[i].operator = rng.gen();
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

    return (child1, child2);
}
