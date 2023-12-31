use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{f64::consts::PI, fs::File, io};

use clap::{ArgAction, Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use fk::ee_pos;
use lgp::{execute, init_registers, num_to_op, CONST_REGS};

use crate::lgp::{
    create_offspring, evaluate_population_par, fitness_uniform_select, init_population, op_to_num,
    tournament_select, Chromosome,
};

mod fk;
mod lgp;

#[derive(clap::ValueEnum, Clone, Copy)]
enum LGPSelection {
    Tournament,
    Fuss,
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
        #[clap(short, long)]
        chromo_max: Option<usize>,
        #[clap(short, long)]
        selection: Option<LGPSelection>,
        #[clap(short, long)]
        meta_iters: Option<usize>,
    },
    IK {
        #[clap(allow_hyphen_values(true))]
        x: f64,
        #[clap(allow_hyphen_values(true))]
        y: f64,
        #[clap(allow_hyphen_values(true))]
        z: f64,
        genome: Option<PathBuf>,
    },
    FK {
        #[clap(allow_hyphen_values(true))]
        th_1: f64,
        #[clap(allow_hyphen_values(true))]
        th_2: f64,
        #[clap(allow_hyphen_values(true))]
        th_3: f64,
    },
    Inspect {
        genome: PathBuf,
    },
}

pub(crate) fn generate_test_angles(
    steps1: usize,
    steps2: usize,
    steps3: usize,
) -> Vec<(f64, f64, f64)> {
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

    output
}

fn smooth_step(x: f64, from: f64, to: f64) -> f64 {
    let out = (x - from) / (to - from);
    out * out * (3.0 - 2.0 * out)
}

fn mean(x: &[(f64, usize)]) -> f64 {
    let mut output = 0.0;
    for (y, _) in x {
        output += y;
    }
    output / (x.len() as f64)
}

fn median(fitness_with_index: &[(f64, usize)]) -> f64 {
    if fitness_with_index.len() % 2 == 1 {
        fitness_with_index[fitness_with_index.len() / 2].0
    } else {
        (fitness_with_index[fitness_with_index.len() / 2].0
            + fitness_with_index[fitness_with_index.len() / 2 - 1].0)
            / 2.0
    }
}

fn train(
    cli_gens: Option<usize>,
    cli_pop: Option<usize>,
    cli_selection: Option<LGPSelection>,
    cli_chromo_max: Option<usize>,
) -> io::Result<f64> {
    let gens = cli_gens.unwrap_or(500);
    let selection = cli_selection.unwrap_or(LGPSelection::Tournament);
    let n = cli_pop.unwrap_or(1000);
    let chromo_max = cli_chromo_max.unwrap_or(100);

    let mut p = init_population(n, 10, chromo_max);

    let test_angles = generate_test_angles(8, 8, 8);
    let n_angles = test_angles.len() as f64;

    // (best, mean, median) fitness
    let mut historical_fitness: Vec<(f64, f64, f64)> = Vec::with_capacity(gens);

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
        fit_with_i.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut new_pop = Vec::with_capacity(n);

        let paired_pop: Vec<(Vec<Chromosome>, Vec<Chromosome>)> = (0..n / 2)
            .into_par_iter()
            .map(|_| {
                let parent1 = match selection {
                    LGPSelection::Tournament => &p[tournament_select(&fit_with_i, 5, 0.8)],
                    LGPSelection::Fuss => &p[fitness_uniform_select(&fit_with_i)],
                };
                let parent2 = match selection {
                    LGPSelection::Tournament => &p[tournament_select(&fit_with_i, 5, 0.8)],
                    LGPSelection::Fuss => &p[fitness_uniform_select(&fit_with_i)],
                };

                create_offspring(
                    parent1,
                    parent2,
                    smooth_step(g as f64 / gens as f64, 0.9, 0.4),
                    smooth_step(g as f64 / gens as f64, 0.8, 0.15),
                    chromo_max,
                )
            })
            .collect();

        for pair in paired_pop {
            new_pop.push(pair.0);
            new_pop.push(pair.1);
        }

        // Elitism
        new_pop[0] = p[fit_with_i[0].1].clone();
        p = new_pop;
        historical_fitness.push((
            fit_with_i[0].0 / n_angles,
            mean(&fit_with_i) / n_angles,
            median(&fit_with_i) / n_angles,
        ));
        bar.set_message(format!(
            "Mean error: {:.5} cm",
            (fit_with_i[0].0 / (test_angles.len() as f64) * (-100.0))
        ));
    }
    bar.finish();

    let mut log_file = File::create("log.txt")?;
    writeln!(log_file, "best_fitness,mean_fitness,median_fitness")?;
    for x in &historical_fitness {
        writeln!(log_file, "{},{},{}", x.0, x.1, x.2)?;
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

    Ok(historical_fitness
        .last()
        .unwrap_or(&(100.0, 100.0, 100.0))
        .0)
}

fn meta_train(
    cli_gens: Option<usize>,
    cli_pop: Option<usize>,
    cli_selection: Option<LGPSelection>,
    cli_chromo_max: Option<usize>,
    meta_iters: usize,
) -> io::Result<f64> {
    let mut best_fitness: f64 = -10000.0;

    for m in 0..meta_iters {
        println!(
            "{}",
            format!("--- Meta Training Iteration {}/{} ---", m + 1, meta_iters).bold()
        );
        let fitness = train(cli_gens, cli_pop, cli_selection, cli_chromo_max)?;
        println!("Resulting best fitness: {:.5} cm. Best fitness so far: {:.5} cm", fitness.abs() * 100.0, best_fitness.abs() * 100.0);

        if fitness > best_fitness {
            fs::copy("log.txt", "meta_best_log.txt")?;
            fs::copy("genome.txt", "meta_best_genome.txt")?;
            best_fitness = fitness;
            println!(
                "{}",
                format!("New fitness found: {:.5} cm", best_fitness.abs() * 100.0).green()
            );
        }

        println!(" ");
    }

    Ok(best_fitness)
}

fn read_genome(path: &Path) -> io::Result<Vec<Chromosome>> {
    let mut individual = vec![];

    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let encoded: Vec<usize> = contents
        .split(", ")
        .map(|x| x.parse().unwrap_or(0))
        .collect::<Vec<usize>>();

    for i in 0..((encoded.len() - 1) / 4) {
        individual.push(Chromosome {
            operator: num_to_op(encoded[i * 4]),
            dest: encoded[i * 4 + 1],
            src1: encoded[i * 4 + 2],
            src2: encoded[i * 4 + 3],
        });
    }

    Ok(individual)
}

fn ik(x: f64, y: f64, z: f64, genome: Option<PathBuf>) -> io::Result<()> {
    let individual = match genome {
        Some(path) => read_genome(&path)?,
        None => init_population(1, 10, 100)[0].clone(),
    };
    let mut regs = init_registers(x, y, z);
    execute(&mut regs, &individual);

    println!(
        "{} {} {}",
        regs[CONST_REGS],
        regs[CONST_REGS + 1],
        regs[CONST_REGS + 2]
    );

    Ok(())
}

fn inspect(genome: PathBuf) -> io::Result<()> {
    let individual = read_genome(&genome)?;

    for chromo in &individual {
        println!("{:?}", chromo);
    }

    println!("Genome consists of: {} chromosomes", individual.len());

    Ok(())
}

fn fk(th_1: f64, th_2: f64, th_3: f64) -> io::Result<()> {
    let (x, y, z) = ee_pos(th_1, th_2, th_3);

    println!("{} {} {}", x, y, z);

    Ok(())
}

fn main() -> io::Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(8).build_global().unwrap();

    let args = Cli::parse();

    match args.command {
        Commands::Train {
            generations,
            population,
            selection,
            chromo_max,
            meta_iters,
        } => {
            match meta_iters {
                Some(iterations) => {
                    meta_train(generations, population, selection, chromo_max, iterations)?
                }
                None => train(generations, population, selection, chromo_max)?,
            };
        }
        Commands::IK { x, y, z, genome } => ik(x, y, z, genome)?,
        Commands::Inspect { genome } => inspect(genome)?,
        Commands::FK { th_1, th_2, th_3 } => fk(th_1, th_2, th_3)?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        lgp::{binary_search, execute, init_registers},
        median,
    };

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

    #[test]
    fn median_selects_correct_entry() {
        let inp1: Vec<(f64, usize)> = (0..10).map(|x| (x as f64, x)).collect();
        let inp2: Vec<(f64, usize)> = (0..11).map(|x| (x as f64, x)).collect();

        assert_eq!(median(&inp1), 4.5, "Even length list");
        assert_eq!(median(&inp2), 5.0, "Odd length list");
    }

    #[test]
    fn execute_identity() {
        let individual = vec![];

        let mut regs = init_registers(0.0, 0.0, 0.0);
        let old_regs = regs.clone();
        execute(&mut regs, &individual);

        for (x, x_p) in old_regs.iter().zip(regs.iter()) {
            assert_eq!(x, x_p);
        }
    }
}
