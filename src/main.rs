use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{f64::consts::PI, fs::File, io};

use clap::{ArgAction, Parser, Subcommand};
use fk::ee_pos;
use indicatif::{ProgressBar, ProgressStyle};
use lgp::{execute, init_registers, num_to_op, CONST_REGS};

use crate::lgp::{
    create_offspring, evaluate_population_par, init_population, op_to_num, tournament_select,
    Chromosome, fitness_uniform_select,
};

mod fk;
mod lgp;

#[derive(clap::ValueEnum, Clone, Copy)]
enum LGPSelection {
    TOURNAMENT,
    FUSS,
}

#[derive(Parser)]
#[command(
    author = "Vincent Udén",
    about = "Calculate joint angles for Hubert using IKs."
)]
struct Cli {
    #[clap(short, long, action=ArgAction::SetTrue)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[clap(short, long)]
        generations: Option<usize>,
        #[clap(short, long)]
        population: Option<usize>,
        selection: Option<LGPSelection>,
    },
    IK {
        x: f64,
        y: f64,
        z: f64,
        genome: Option<PathBuf>,
    },
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

fn smooth_step(x: f64, from: f64, to: f64) -> f64 {
    let out = (x - from) / (to - from);
    return out * out * (3.0 - 2.0 * out);
}

fn train(cli_gens: Option<usize>, cli_pop: Option<usize>, cli_selection: Option<LGPSelection>) -> io::Result<()> {
    let gens = cli_gens.unwrap_or(500);
    let selection = cli_selection.unwrap_or(LGPSelection::TOURNAMENT);
    let n = cli_pop.unwrap_or(1000);

    let mut p = init_population(n, 10, 100);

    let test_angles = generate_test_angles(10, 10, 10);

    let mut historical_fitness = Vec::with_capacity(gens);

    let bar = ProgressBar::new(gens as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}/{duration_precise}] {bar:40.cyan/blue} {pos:>5}/{len:5} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    for g in 0..gens {
        bar.inc(1);
        let fitness = evaluate_population_par(&p, &test_angles);
        let mut fit_with_i: Vec<(f64, usize)> = fitness.into_iter().zip(0..n).collect();
        fit_with_i.sort_by(|a, b| {
            return b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal);
        });

        let mut new_pop = Vec::with_capacity(n);

        let paired_pop: Vec<(Vec<Chromosome>, Vec<Chromosome>)> = (0..n / 2)
            //.into_par_iter()
            .map(|_| {
                let parent1 = match selection {
                    LGPSelection::TOURNAMENT => &p[tournament_select(&fit_with_i, 5, 0.8)],
                    LGPSelection::FUSS => &p[fitness_uniform_select(&fit_with_i)],
                };
                let parent2 = match selection {
                    LGPSelection::TOURNAMENT => &p[tournament_select(&fit_with_i, 5, 0.8)],
                    LGPSelection::FUSS => &p[fitness_uniform_select(&fit_with_i)],
                };

                return create_offspring(
                    parent1,
                    parent2,
                    smooth_step(g as f64 / gens as f64, 0.9, 0.25),
                    smooth_step(g as f64 / gens as f64, 0.8, 0.05),
                );
            })
            .collect();

        for pair in paired_pop {
            new_pop.push(pair.0);
            new_pop.push(pair.1);
        }

        new_pop[0] = p[fit_with_i[0].1].clone();
        p = new_pop;
        historical_fitness.push(fit_with_i[0].0);
        bar.set_message(format!(
            "Mean error: {:.5} cm",
            (fit_with_i[0].0 / (test_angles.len() as f64) * (-100.0))
        ));
    }
    bar.finish();

    let mut log_file = File::create("log.txt")?;
    for x in &historical_fitness {
        write!(log_file, "{},", x)?;
    }
    writeln!(log_file)?;

    let mut genome_file = File::create("genome.txt")?;
    for x in &p[0] {
        write!(
            genome_file,
            "{}, {}, {}, {}, ",
            op_to_num(x.operator),
            x.dest,
            x.src1,
            x.src2
        )?;
    }
    writeln!(genome_file)?;

    return Ok(());
}

fn read_genome(path: &Path) -> io::Result<Vec<Chromosome>> {
    let mut individual = vec![];

    println!("{}", path.to_str().unwrap());
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let encoded: Vec<usize> = contents
        .split(", ")
        .map(|x| {
            println!("{} {}", x, x.parse::<i32>().unwrap_or(0));
            x.parse().unwrap_or(0)
        })
        .collect::<Vec<usize>>();

    for i in 0..((encoded.len() - 1) / 4) {
        println!(
            "{} {} {} {}",
            encoded[i * 4],
            encoded[i * 4 + 1],
            encoded[i * 4 + 2],
            encoded[i * 4 + 3]
        );
        individual.push(Chromosome {
            operator: num_to_op(encoded[i * 4]),
            dest: encoded[i * 4 + 1],
            src1: encoded[i * 4 + 2],
            src2: encoded[i * 4 + 3],
        });
    }

    for x in &individual {
        println!("{:?}", x);
    }

    return Ok(individual);
}

fn ik(x: f64, y: f64, z: f64, genome: Option<PathBuf>) -> io::Result<()> {
    let individual = match genome {
        Some(path) => read_genome(&path)?,
        None => init_population(1, 10, 100)[0].clone(),
    };
    let mut regs = init_registers(x, y, z);
    execute(&mut regs, &individual);

    let (x_p, y_p, z_p) = ee_pos(
        regs[CONST_REGS + 0],
        regs[CONST_REGS + 1],
        regs[CONST_REGS + 2],
    );

    println!("{} {} {}", x_p, y_p, z_p);

    return Ok(());
}

fn main() -> io::Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Train { generations, population, selection } => train(generations, population, selection)?,
        Commands::IK { x, y, z, genome } => ik(x, y, z, genome)?,
    }

    return Ok(());
}

#[cfg(test)]
mod tests {
    use crate::lgp::binary_search;

    #[test]
    fn binary_search_finds_closest_match() {
        let mut inp: Vec<(f64, usize)> = vec![];
        for x in 0..10 {
            inp.push((1.0 - (x as f64 / 10.0), x));
        }

        assert_eq!(binary_search(&inp, 0.0).1, 9);
        assert_eq!(binary_search(&inp, 10.0).1, 0);
        assert_eq!(binary_search(&inp, 0.2).1, 8);
        assert_eq!(binary_search(&inp, 0.96).1, 0);
        assert_eq!(binary_search(&inp, 0.94).1, 1);
    }
}
