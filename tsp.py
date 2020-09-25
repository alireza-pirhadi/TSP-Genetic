import random
import math
import matplotlib.pyplot as plt
import numpy as np


def evaluate_chromosome(chromosome):
    fitness = distances[chromosome[:], np.concatenate((chromosome[1:], chromosome[0:1]))].sum()
    return math.pow(10,5) / fitness


# PMX
def cross_over(first_chromosome, second_chromosome):
    # first_point = int(random.random() * number_of_cities)
    # second_point = int(random.random() * number_of_cities)
    # while first_point >= second_point:
    #     first_point = int(random.random() * number_of_cities)
    #     second_point = int(random.random() * number_of_cities)
    first_point = int(random.random() * 15)
    second_point = int(179 + random.random() * 15)
    selected_part_1 = first_chromosome[first_point:second_point]
    selected_part_2 = second_chromosome[first_point:second_point]
    uncommons_1 = []
    uncommons_2 = []
    for i in range(len(selected_part_1)):
        if not(selected_part_1[i] in selected_part_2):
            uncommons_1.append(selected_part_1[i])
    for i in range(len(selected_part_2)):
        if not(selected_part_2[i] in selected_part_1):
            uncommons_2.append(selected_part_2[i])
    res = []
    first_child = []
    second_child = []
    for i in range(first_point):
        first_child.append(first_chromosome[i])
        second_child.append(second_chromosome[i])
    for i in range(second_point - first_point):
        first_child.append(selected_part_2[i])
        second_child.append(selected_part_1[i])
    for i in range(number_of_cities - second_point):
        first_child.append(first_chromosome[second_point + i])
        second_child.append(second_chromosome[second_point + i])

    index1 = 0
    index2 = 0
    for i in range(number_of_cities):
        if first_child.count(first_child[i]) > 1:
            first_child[i] = uncommons_1[index1]
            index1 += 1
        if second_child.count(second_child[i]) > 1:
            second_child[i] = uncommons_2[index2]
            index2 += 1

    res.append(first_child)
    res.append(second_child)
    return res


# Reverse mutation
def mutate(ch):
    if random.random() < mutation_rate:
        # SHUFFLE
        # ch.clear()
        # for j in range(number_of_cities):
        #     ch.append(0)
        # for j in range(1, number_of_cities + 1):
        #     rand = int(random.random() * number_of_cities)
        #     while ch[rand] != 0:
        #         rand = int(random.random() * number_of_cities)
        #     ch[rand] = j
        first_point = int(random.random() * number_of_cities)
        second_point = int(random.random() * number_of_cities)
        while first_point >= second_point:
            first_point = int(random.random() * number_of_cities)
            second_point = int(random.random() * number_of_cities)
        selected_part = ch[first_point:second_point]
        for i in range(len(selected_part)):
            ch[first_point+i] = selected_part[len(selected_part) - i - 1]
        # CHANGING POSITION OF CITIES
        # for i in range(1+15*int(generation_count/1000)):
        #     first_city = int(random.random() * number_of_cities)
        #     second_city = int(random.random() * number_of_cities)
        #     while first_city == second_city:
        #         second_city = int(random.random() * number_of_cities)
        #     tmp = ch[first_city]
        #     ch[first_city] = ch[second_city]
        #     ch[second_city] = tmp


def sort_population(p):
    for i in range(population_size):
        for j in range(i + 1, len(p)):
            if evaluate_chromosome(p[i]) < evaluate_chromosome(p[j]):
                tmp = p[i]
                p[i] = p[j]
                p[j] = tmp


def min(p):
    min_value = -1
    for i in range(len(p)):
        if min_value > evaluate_chromosome(p[i]) or min_value == -1:
            min_value = evaluate_chromosome(p[i])
    return min_value


def max(p):
    max_value = -1
    for i in range(len(p)):
        if max_value < evaluate_chromosome(p[i]):
            max_value = evaluate_chromosome(p[i])
    return max_value


# read data from file
best_values = []
worst_values = []
avg_values = []
f = open("tsp_data.txt", "r")
number_of_cities = 194
cities = []
for i in range(number_of_cities):
    line = f.readline().split(" ")
    cities.append([float(line[1]), float(line[2])])
distances = np.zeros((number_of_cities+1, number_of_cities+1))
for i in range(1, number_of_cities+1):
    for j in range(1, number_of_cities+1):
        distances[i][j] = math.sqrt(((cities[i-1][0] - cities[j-1][0]) ** 2) + ((cities[i-1][1] - cities[j-1][1]) ** 2))


# initialize population
number_of_generations = 5000
generation_count = 1
population_size = 50
population = []
for i in range(population_size):
    while True:
        new_chromosome = []
        for j in range(number_of_cities):
            new_chromosome.append(0)
        for j in range(1, number_of_cities + 1):
            rand = int(random.random() * number_of_cities)
            while new_chromosome[rand] != 0:
                rand = int(random.random() * number_of_cities)
            new_chromosome[rand] = j
        if not(new_chromosome in population):
            population.append(new_chromosome)
            break


number_of_parents = 12
tournament_size = 10
remaining = []
while generation_count <= number_of_generations:
    # choosing parents with q tournament ( q = 10 )
    tmp_population = population.copy()
    evaluations = []
    for i in range(population_size):
        evaluations.append(evaluate_chromosome(population[i]))
    parents = []
    for i in range(number_of_parents):
        randoms = []
        for j in range(tournament_size):
            r = int(random.random() * len(tmp_population))
            while r in randoms:
                r = int(random.random() * len(tmp_population))
            randoms.append(r)
        chosen_parent = randoms[0]
        for j in range(1, tournament_size):
            tmp = randoms[j]
            if evaluations[chosen_parent] < evaluations[tmp]:
                chosen_parent = tmp
        parents.append(tmp_population[chosen_parent])
        tmp_population.remove(tmp_population[chosen_parent])
        evaluations.remove(evaluations[chosen_parent])
    # parents = []
    # for i in range(number_of_parents):
    #     for j in range(i + 1, population_size):
    #         if evaluate_chromosome(population[i]) < evaluate_chromosome(population[j]):
    #             tmp = population[i]
    #             population[i] = population[j]
    #             population[j] = tmp
    #     parents.append(population[i])


    # cross-over and mutate
    mutation_rate = 0.5
    parents_and_children = population.copy()
    for i in range(int(population_size / 2)):
        first = int(random.random()*number_of_parents)
        second = int(random.random()*number_of_parents)
        while first == second:
            second = int(random.random() * number_of_parents)
        children = cross_over(parents[first], parents[second])
        mutate(children[0])
        mutate(children[1])
        parents_and_children.append(children[0])
        parents_and_children.append(children[1])


    # remaining selection
    probabilities = []
    remaining.clear()
    sum_probabilities = 0.0
    for i in range(len(parents_and_children)):
        sum_probabilities += evaluate_chromosome(parents_and_children[i])
    probabilities.append(evaluate_chromosome(parents_and_children[0]) / sum_probabilities)
    for i in range(1, len(parents_and_children)):
        probabilities.append(probabilities[i - 1] + (evaluate_chromosome(parents_and_children[i]) / sum_probabilities))
    for i in range(population_size):
        rand = random.random()
        for j in range(len(parents_and_children)):
            if rand < probabilities[j]:
                remaining.append(parents_and_children[j])
                break
        #rand += (1 / population_size)
    # sort_population(parents_and_children)
    # remaining = []
    # for i in range(population_size):
    #     remaining.append(parents_and_children[i])


    print("Best : ", math.pow(10,5) / max(remaining), ",Worst : ", math.pow(10,5) / min(remaining))
    best_values.append(math.pow(10,5) / max(remaining))
    worst_values.append(math.pow(10,5) / min(remaining))
    avg = 0
    for i in range(population_size):
        avg += evaluate_chromosome(remaining[i])
    avg /= population_size
    avg_values.append(math.pow(10,5) / avg)
    population = remaining
    generation_count += 1

x = []
for i in range(number_of_generations):
    x.append(i)
# plotting the line 1 points
plt.plot(x, best_values, label="best")

# plotting the line 2 points
plt.plot(x, avg_values, label="average")

# plotting the line 2 points
plt.plot(x, worst_values, label="worst")

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('path length in generations')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()

result = remaining[0]
for i in range(1, len(remaining)):
    if evaluate_chromosome(result) < evaluate_chromosome(remaining[i]):
        result = remaining[i]
x1 = []
x2 = []
for i in range(number_of_cities):
    x1.append(cities[result[i]-1][0])
    x2.append(cities[result[i]-1][1])
x1.append(cities[result[0]-1][0])
x2.append(cities[result[0]-1][1])
plt.plot(x1, x2, label="path")
plt.legend()
plt.show()