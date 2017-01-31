#!/usr/bin/env python3

## Joseph McGill
## Genetic Algorithms
## Function maximization f(x, y)
##
## Genetic algorithm implementation for function maximization. The GA
## parameters are defined below. This GA uses uniform crossover and tournament
## selection as genetic operators. The function this GA approximates is
## f(x, y) = sin(x) x J_1(y); 0 <= x, y <= 5; where J_1 is the Bessel function
## of the first kind.

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# Constants for the GA
NUM_DEC_PLACES = 4
POP_SIZE = 40
MAX_GENS = 100
CROSSOVER_PROB = 0.6
MUTATION_PROB = 0.25
TOURNAMENT_SIZE = 2

# Function to create a population
def generate_pop():

    population = []

    for pop in range(POP_SIZE):
        x_vals = []
        x_vals.append(np.random.randint(X_RANGE[0], X_RANGE[1] + 1))

        # if the int generated is the max number, fill the rest with 0s
        if (x_vals[0] == X_RANGE[0] or x_vals[0] == X_RANGE[1]):
            for i in range(NUM_DEC_PLACES):
                x_vals.append(0)

        else:
            for i in range(NUM_DEC_PLACES):
                x_vals.append(np.random.randint(0, 10))


        y_vals = []
        y_vals.append(np.random.randint(Y_RANGE[0], Y_RANGE[1] + 1))

        # if the int generated is the max number, fill the rest with 0s
        if (y_vals[0] == Y_RANGE[0] or y_vals[0] == Y_RANGE[1]):
            for i in range(NUM_DEC_PLACES):
                y_vals.append(0)

        else:
            for i in range(NUM_DEC_PLACES):
                y_vals.append(np.random.randint(0, 10))


        population.append(x_vals + y_vals)

    # return the population
    return population

# Function to generate the next generation
def next_gen(population):

    # determine the number of crossovers and clones
    cross_num = int(POP_SIZE * CROSSOVER_PROB)
    clone_num = POP_SIZE - cross_num

    # create part of the  next generation with crossover
    next_gen = []
    for i in range(0, cross_num, 2):
        child_1, child_2 = crossover(population)
        next_gen.append(child_1)
        next_gen.append(child_2)

    # create part of the next generation with selection
    for i in range(clone_num): next_gen.append(selection(population))

    # mutate the next generation
    next_gen = mutation(next_gen)

    # return the next generation
    return next_gen

# Function to calculate the fitness of an individual
def fitness(individual):

    #convert the individuals into x and y
    #using the number of decimal places defined in the population
    x = float(''.join(map(str,
              individual[:(NUM_DEC_PLACES + 1)])))/10**NUM_DEC_PLACES
    y = float(''.join(map(str,
              individual[(NUM_DEC_PLACES + 1):])))/10**NUM_DEC_PLACES

    # return the fitness of the individual
    return np.sin(x) + (2 * special.jv(1, y))

# Function to perform tournament selection selection
def selection(population):

    # get random indices for use in the tournament
    ind = np.random.choice(range(POP_SIZE), TOURNAMENT_SIZE)

    # create the tournament round
    tournament = []
    for index in ind: tournament.append((index, fitness(population[index])))

    # return the tournament winner
    return population[max(tournament, key=lambda x:x[1])[0]]

# Function to perform crossover
def crossover(population):

    # get the 2 parents for crossover
    parent_1 = selection(population)
    parent_2 = selection(population)

    child_1 = []
    child_2 = []

    # perform crossover
    for i in range(len(parent_1)):

        # if 1, child_1's gene comes from parent_1, else parent_2
        # uniform crossover with a probability of 0.5
        if (np.random.binomial(1, 0.5, 1) == 0):
            child_1.append(parent_1[i])
            child_2.append(parent_2[i])

        else:
            child_1.append(parent_2[i])
            child_2.append(parent_1[i])

    # extract the x and y values for child_1 to bound it
    c1_x = float(''.join(map(str,
            child_1[:(NUM_DEC_PLACES + 1)])))/10**NUM_DEC_PLACES
    c1_y = float(''.join(map(str,
            child_1[(NUM_DEC_PLACES + 1):]))) /10**NUM_DEC_PLACES

    # if the mutation causes x or y to be greater (or less) than their max (min)
    # ignore the mutation and return the original individual
    if (c1_x < X_RANGE[0]):
        child_1[0] = X_RANGE[0] + 1

    elif (c1_x > X_RANGE[1]):
        child_1[0] = X_RANGE[1] - 1

    elif (c1_y < Y_RANGE[0]):
        child_1[NUM_DEC_PLACES + 1] = Y_RANGE[0] + 1

    elif (c1_y > Y_RANGE[1]):
        child_1[NUM_DEC_PLACES + 1] = Y_RANGE[1] - 1


    # extract the x and y values for child_1 to bound it
    c2_x = float(''.join(map(str,
            child_2[:(NUM_DEC_PLACES + 1)])))/10**NUM_DEC_PLACES
    c2_y = float(''.join(map(str,
            child_2[(NUM_DEC_PLACES + 1):])))/10**NUM_DEC_PLACES

    # if the mutation causes x or y to be greater (or less) than their
    # max (min) ignore the mutation and return the original individual
    if (c2_x < X_RANGE[0]):
        child_2[0] = X_RANGE[0] + 1

    elif (c2_x > X_RANGE[1]):
        child_2[0] = X_RANGE[1] - 1

    elif (c2_y < Y_RANGE[0]):
        child_2[NUM_DEC_PLACES + 1] = Y_RANGE[0] + 1

    elif (c2_y > Y_RANGE[1]):
        child_2[NUM_DEC_PLACES + 1] = Y_RANGE[1] - 1

    # return the bounded offspring
    return child_1, child_2

# Function to mutate a population
def mutation(population):

    mutations = np.random.binomial(1, MUTATION_PROB, len(population))
    mutated_pop = []
    for index, mutate in enumerate(mutations):

        # mutate (if 1) by randomly selecting a number
        # and replacing it with a random int between 0 and 9
        if (mutate == 1):

            # mutate a random digit
            i = np.random.randint(0, len(population[index]))
            mutated_individual = population[index]
            mutated_individual[i] = np.random.randint(0, 10)

            # extract the x and y values
            x = float(''.join(map(str, mutated_individual[:(NUM_DEC_PLACES
                                               + 1)])))/10**NUM_DEC_PLACES

            y = float(''.join(map(str, mutated_individual[(NUM_DEC_PLACES
                                               + 1):])))/10**NUM_DEC_PLACES

            # if the mutation causes x or y to be greater (or less) than
            # their max (min) ignore the mutation and return the original
            # individual
            if (x < X_RANGE[0] or x > X_RANGE[1]
            or y < Y_RANGE[0] or y > Y_RANGE[1]):
                mutated_pop.append(population[index])
            else:
                mutated_pop.append(mutated_individual)

        else:
            mutated_pop.append(population[index])

    # return the mutated population
    return mutated_pop


# the bounds for the function
X_RANGE = [0, 5]
Y_RANGE = [0, 5]

# Run the GA 100 times for statistics
solutions = []
data = []
for i in range(100):

    row = []
    row.append(i + 1)

    # generate the initial population
    new_pop = generate_pop()

    # find the max fitness of the population
    fit = [(indiv, fitness(indiv), 0) for indiv in new_pop]
    max_fitness = max(fit, key=lambda x:x[1])

    avg_fit = []
    # evolve the population MAX_GENS times
    for generation in range(MAX_GENS):

        # generate the next population
        new_pop = next_gen(new_pop)

        # find the max fitness
        fit = [(indiv, fitness(indiv), generation) for indiv in new_pop]
        pop_fit = [fitness(indiv) for indiv in new_pop]

        gen_max = max(fit, key=lambda x:x[1])

        # update the current max if necesssary
        if (gen_max[1] > max_fitness[1]):
            max_fitness = gen_max

        avg_fit.append(np.mean(pop_fit))

    # add the current max to the solutions
    solutions.append(max_fitness)
    row.append(max_fitness[1])
    row.append(max_fitness[2])

    # add the average population fitnesses to the list
    row.append(avg_fit[0])
    row.append(avg_fit[24])
    row.append(avg_fit[49])
    row.append(avg_fit[74])
    row.append(avg_fit[99])

    # add the data row to the dataset
    data.append(row)

###### Output ######
print("###### OUTPUT ######")
print("x bounds: [%d, %d]" % (X_RANGE[0], X_RANGE[1]))
print("y bounds: [%d, %d]" % (Y_RANGE[0], Y_RANGE[1]))

# optimal solution using 4 decimal places
print("Optimal solution: %.10f" % (np.sin(1.5708) + 2*special.jv(1, 1.841)))
optimal = np.round(np.sin(1.5708) + 2*special.jv(1, 1.841), 4)


# count the number of correctly found solutions
rounded = [np.round(sol[1], 4) for sol in solutions]
correct = 0
for i in solutions:
    if (abs(optimal - i[1]) < 0.1):
        correct += 1

print("Percent of global maximums found: %.2f" % correct)

# average the solutions and number of generations taken
max_fit = [x[1] for x in solutions]
generations = [x[2] for x in solutions]
print("Mean max fitness: %.10f" % np.mean(max_fit))
print("Mean generations: %.4f\n" % np.mean(generations))

# plot the max fitnesses found
plt.plot(range(0, 100), rounded, 'bo')
plt.plot([0, 100], [np.round(optimal, 4), np.round(optimal, 4)], 'r', lw=2)
plt.ylim([0, 2.5])
plt.xlabel("Runs")
plt.ylabel("Max fitness")
plt.title("Function 1 Maximization using Genetic Algorithm")
plt.show()
