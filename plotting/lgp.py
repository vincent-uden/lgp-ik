import numpy as np
import copy
import itertools

from typing import List
from random import random, randrange
from eqns import A_3_to_0

def add(x, y):
    return x + y

def sub(x, y):
    return x - y
def mul(x, y):
    return x * y

def div(x, y):
    if y == 0:
        y = 0.001
    return x / y

def sin(x, y):
    return np.sin(x)

def cos(x, y):
    return np.cos(x)

def sqrt(x, y):
    return np.sqrt(np.abs(x))

operators = [
    add,
    sub,
    mul,
    div,
    sin,
    cos,
    sqrt,
]

op_names = [
    "add",
    "sub",
    "mul",
    "div",
    "sin",
    "cos",
    "sqrt",
]

CONST_REGS = [1.0, 0.5]
VAR_REGS_N = 5

def execute(registers: List[float], genome: List[int]):
    # genome expected to be of shape (N*4) where N is the amount of
    # chromosomes in that individual
    for i in range(len(genome) // 4):
        op, dest, op1, op2 = genome[i*4:i*4+4]
        # print(op, dest, op1, op2)
        registers[dest] = operators[op](registers[op1], registers[op2])

def print_genome(genome: List[int]):
    output = ""
    for i in range(len(genome) // 4):
        op, dest, op1, op2 = genome[i*4:i*4+4]
        output += f"{op_names[op]}({op1},{op2}) -> {dest}\n"
    print(output)

def init_registers(x, y, z) -> np.ndarray:
    # The two first positions are constant registers
    # We provide input through the three variable registers
    return np.array(CONST_REGS + [x, y, z] + [0.0] * (VAR_REGS_N - 3))

def clamp(x, lower, upper):
    return min(upper, max(x, lower))

def valid_operator(x=None) -> int:
    if x == None:
        x = randrange(0, len(operators))
    return clamp(x, 0, len(operators))

def valid_dest_register(x=None):
    if x == None:
        x = randrange(len(CONST_REGS), len(CONST_REGS) + VAR_REGS_N)
    return clamp(x, len(CONST_REGS), len(CONST_REGS) + VAR_REGS_N)

def valid_src_register(x=None):
    if x == None:
        x = randrange(0, len(CONST_REGS) + VAR_REGS_N)
    return clamp(x, 0, len(CONST_REGS) + VAR_REGS_N)

def init_population(N: int, min_chromo: int, max_chromo: int) -> List[List[int]]:
    population = []
    for i in range(N):
        individual = []
        chromos = randrange(min_chromo, max_chromo)
        for j in range(chromos):
            individual.append(valid_operator())
            individual.append(valid_dest_register())
            individual.append(valid_src_register())
            individual.append(valid_src_register())
        population.append(individual)
    return population

def evaluate_population(population: List[List[int]]) -> List[float]:
    # test_angles = np.random.rand(3, 1000) * np.pi
    # test_angles[2,:] /= 2
    test_angles_1 = np.linspace(0, np.pi, num=10)
    test_angles_2 = np.linspace(0, np.pi, num=10)
    test_angles_3 = np.linspace(0, np.pi/2, num=10)
    test_angles = np.array(list(itertools.product(test_angles_1, test_angles_2, test_angles_3))).T

    fitness = []

    for j, individual in enumerate(population):
        error = 0.0
        for i in range(test_angles.shape[1]):
            x, y, z, _ = A_3_to_0(*test_angles[:,i], np.zeros(3))
            regs = init_registers(x, y, z)
            execute(regs, individual)
            x_p, y_p, z_p, _ = A_3_to_0(*regs[len(CONST_REGS):len(CONST_REGS)+3], np.zeros(3))
            # if j == 0:
                # print(x, y, z)
                # print(regs[len(CONST_REGS):])
                # print(x_p, y_p, z_p)
                # print("")
            error += np.power(x - x_p, 2)
            error += np.power(y - y_p, 2)
            error += np.power(z - z_p, 2)
        fitness.append(-error)

    return fitness

def tournament_selection(fitness: List[float], t_size: int = 2, p_tour: float = 0.7) -> int:
    indices = [ (randrange(0, len(fitness)), i) for i in fitness ]
    ordered = sorted(indices, key=lambda x: x[1], reverse=True)
    i = 0
    while random() > p_tour and i < len(ordered):
        i += 1
    return int(ordered[i][0])

def create_offspring(parent1: List[int], parent2: List[int], p_cross: float = 0.2, p_mut: float = 0.1) -> (List[int], List[int]):
    MAX_LEN = 4 * 128
    child1 = parent1[::]
    child2 = parent2[::]

    if random() < p_cross and len(child1) > 4 and len(child2) > 4:
        # We don't want potentially empty children
        cross_point_1_1 = randrange(0, len(child1) // 4 - 1)
        cross_point_1_2 = randrange(cross_point_1_1 + 1, len(child1) // 4)
        cross_point_2_1 = randrange(0, len(child2) // 4 - 1)
        cross_point_2_2 = randrange(cross_point_2_1 + 1, len(child2) // 4)
        tmp1 = child1[:cross_point_1_1*4] + child2[cross_point_2_1*4:cross_point_2_2*4] + child1[cross_point_1_2*4:]
        tmp2 = child2[:cross_point_2_1*4] + child1[cross_point_1_1*4:cross_point_1_2*4] + child2[cross_point_2_2*4:]

        child1 = tmp1[:MAX_LEN]
        child2 = tmp2[:MAX_LEN]

    for i in range(len(child1)):
        if random() < p_mut:
            if i % 4 == 0:
                child1[i] = valid_operator()
            if i % 4 == 1:
                child1[i] = valid_dest_register()
            if i % 4 == 2:
                child1[i] = valid_src_register()
            if i % 4 == 3:
                child1[i] = valid_src_register()
    for i in range(len(child2)):
        if random() < p_mut:
            if i % 4 == 0:
                child2[i] = valid_operator()
            if i % 4 == 1:
                child2[i] = valid_dest_register()
            if i % 4 == 2:
                child2[i] = valid_src_register()
            if i % 4 == 3:
                child2[i] = valid_src_register()
    return (child1, child2)

if __name__ == "__main__":
    GENERATIONS = 1000
    N = 100
    p = init_population(N, 10, 100)
    historical_max_fitness = []
    for g in range(GENERATIONS):
        fitness = evaluate_population(p)
        new_pop = []
        avg_len = 0.0
        for i in range(len(p) // 2):
            j = tournament_selection(fitness, 5, 0.7)
            parent1 = p[j][:]
            j = tournament_selection(fitness, 5, 0.7)
            parent2 = p[j][:]

            # A typical mutation rate is roughly 1/chromosome_length
            child1, child2 = create_offspring(parent1, parent2, 0.25, 5.0/50)
            new_pop.append(child1)
            new_pop.append(child2)

            avg_len += len(child1) / float(N)
            avg_len += len(child2) / float(N)

        new_pop[0] = p[np.argmax(fitness)]
        p = copy.deepcopy(new_pop)
        historical_max_fitness.append(np.max(fitness))
        print(f"Gen: {g+1}/{GENERATIONS} Max fitness: {np.max(fitness)}, Best genome length: {len(p[0])}, Avg genome length: {avg_len}")

    with open("./log.txt", "w") as f:
        for x in historical_max_fitness:
            f.write(f"{x}, ")
        f.write("\n")

    with open("./genome.txt", "w") as f:
        for x in p[np.argmax(fitness)]:
            f.write(f"{x}, ")
        f.write("\n")

    print(p[np.argmax(fitness)])
