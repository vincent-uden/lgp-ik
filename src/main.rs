use std::{f64::consts::PI, fs::File, io};
use std::io::{Write};

use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};

use crate::fk::ee_pos;

mod fk;

#[derive(Debug, Clone, Copy)]
enum Op {
    ADD,
    SUB,
    MUL,
    DIV,
    SIN,
    COS,
    SQT,
    ASIN,
}

impl Distribution<Op> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Op {
        match rng.gen_range(0..7) {
            0 => Op::ADD,
            1 => Op::SUB,
            2 => Op::MUL,
            3 => Op::DIV,
            4 => Op::SIN,
            5 => Op::COS,
            6 => Op::SQT,
            7 => Op::ASIN,
            _ => Op::ADD,
        }
    }
}

fn op_to_num(op: Op) -> usize {
    return match op {
        Op::ADD => 0,
        Op::SUB => 1,
        Op::MUL => 2,
        Op::DIV => 3,
        Op::SIN => 4,
        Op::COS => 5,
        Op::SQT => 6,
        Op::ASIN => 7,
    };
}

static CONST_REGS: usize = 2;
static VAR_REGS: usize = 5;

#[derive(Debug, Clone, Copy)]
struct Chromosome {
    operator: Op,
    dest: usize,
    src1: usize,
    src2: usize,
}

fn safe_div(x: f64, y: f64) -> f64 {
    let divisor: f64 = if y == 0.0 { 0.0000001 } else { y };

    return x / divisor;
}

fn execute(regs: &mut [f64], genome: &[Chromosome]) {
    for chromo in genome {
        regs[chromo.dest] = match chromo.operator {
            Op::ADD => regs[chromo.src1] + regs[chromo.src2],
            Op::SUB => regs[chromo.src1] - regs[chromo.src2],
            Op::MUL => regs[chromo.src1] * regs[chromo.src2],
            Op::DIV => safe_div(regs[chromo.src1], regs[chromo.src2]),
            Op::SIN => regs[chromo.src1].sin(),
            Op::COS => regs[chromo.src1].cos(),
            Op::SQT => regs[chromo.src1].abs().sqrt(),
            Op::ASIN => regs[chromo.src1].clamp(-1.0, 1.0).asin(),
        }
    }
}

fn init_registers(th_1: f64, th_2: f64, th_3: f64) -> Vec<f64> {
    let mut output = Vec::with_capacity(CONST_REGS + VAR_REGS);
    output.push(1.0);
    output.push(0.5);
    output.push(th_1);
    output.push(th_2);
    output.push(th_3);
    for _ in 5..(CONST_REGS + VAR_REGS) {
        output.push(0.0);
    }
    return output;
}

fn init_population(n: usize, min_chromo: usize, max_chromo: usize) -> Vec<Vec<Chromosome>> {
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

fn evaluate_population(
    population: &Vec<Vec<Chromosome>>,
    test_angles: &[(f64, f64, f64)],
) -> Vec<f64> {
    let mut fitness = Vec::with_capacity(population.len());

    let mut i = 0;
    for individual in population {
        let mut error = 0.0;
        for (th_1, th_2, th_3) in test_angles {
            let (x, y, z) = ee_pos(*th_1, *th_2, *th_3);

            let mut regs = init_registers(x, y, z);
            execute(&mut regs, individual);

            let (x_p, y_p, z_p) = ee_pos(regs[2], regs[3], regs[4]);
            if i == 0 && false {
                println!("Real: {} {} {}", x, y, z);
                println!("Angl: {} {} {}", regs[2], regs[3], regs[4]);
                println!("Pred: {} {} {}", x_p, y_p, z_p);
            }

            error += (x - x_p).powi(2) + (y - y_p).powi(2) + (z - z_p).powi(2);
        }
        i += 1;
        fitness.push(-error);
    }

    return fitness;
}

fn generate_test_angles(steps1: usize, steps2: usize, steps3: usize) -> Vec<(f64, f64, f64)> {
    let mut output = Vec::with_capacity(steps1 * steps2 * steps3);

    for i in 0..steps1 {
        for j in 0..steps2 {
            for k in 0..steps3 {
                output.push((
                    PI / (steps1 as f64) * (i as f64),
                    PI / (steps2 as f64) * (j as f64),
                    0.5 * PI / (steps3 as f64) * (k as f64),
                ));
            }
        }
    }

    return output;
}

fn tournament_select(sorted_fitness_with_i: &Vec<(f64, usize)>, t_size: usize, p_tour: f64) -> usize {
    let mut i = 0;
    while thread_rng().gen::<f64>() < p_tour && i < t_size && i < sorted_fitness_with_i.len() {
        i += 1;
    }
    return sorted_fitness_with_i[i].1;
}

fn create_offspring(
    parent1: &Vec<Chromosome>,
    parent2: &Vec<Chromosome>,
    p_cross: f64,
    p_mut: f64,
) -> (Vec<Chromosome>, Vec<Chromosome>) {
    const MAX_LEN: usize = 200;
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

fn main() -> io::Result<()> {
    const GENERATIONS: usize = 100000;
    const N: usize = 100;
    let mut p = init_population(N, 10, 100);

    let test_angles = generate_test_angles(10, 10, 10);

    let mut historical_fitness = Vec::with_capacity(GENERATIONS);

    for g in 0..GENERATIONS {
        let fitness = evaluate_population(&p, &test_angles);
        let mut fit_with_i: Vec<(f64, usize)> = fitness.into_iter().zip(0..N).collect();
        fit_with_i.sort_by(|a, b| {
            return b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal);
        });

        let mut new_pop = Vec::with_capacity(N);

        for _ in 0..N/2 {
            let parent1 = &p[tournament_select(&fit_with_i, 5, 0.7)];
            let parent2 = &p[tournament_select(&fit_with_i, 5, 0.7)];

            let (child1, child2) = create_offspring(parent1, parent2, 0.25, 0.1);

            new_pop.push(child1);
            new_pop.push(child2);
        }

        new_pop[0] = p[fit_with_i[0].1].clone();
        p = new_pop;
        historical_fitness.push(fit_with_i[0].0);
        println!("Gen: {}/{} Max fitness: {}", g+1, GENERATIONS, fit_with_i[0].0);
    }

    let mut log_file = File::create("log.txt")?;
    for x in &historical_fitness {
        write!(log_file, "{},", x)?;
    }
    writeln!(log_file)?;

    let mut genome_file = File::create("genome.txt")?;
    for x in &p[0] {
        write!(genome_file, "{}, {}, {}, {}, ", op_to_num(x.operator), x.dest, x.src1, x.src2)?;
    }
    writeln!(genome_file)?;

    return Ok(());
}
