#!/usr/bin/env python3


# Imports
import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from game import Game

from players import AI

PIECELIMIT = -1 # Maximum number of pieces in a game before game over. Set to -1 for unlimited

########################
# HYPER PARAMETERS
########################
P_MUTATION = 0.1 # mutation probability
P_CROSSOVER = 0.5 # crossover probability

"""
########################
# SIMPLE EA FROM SLIDES
########################
Steps (you can see them back in the code below):
1: Init population with candidate solutions
2: Evaluate quality of each candidate
3: Repeat until a termination condition is satisfied:
    a: Select candidate solutions for reproduction
    b: Recombine selected candidates
    c: Mutate the resulting candidates
    d: Evaluate the new candidates
    e: Select candidates for the next generation

"""


class SimpleEA:

    # constructor
    def __init__(self, weights, popsize = 50, poffspring = 0.7, pmut = 0.1, termiterations = 10):
        self.weights = weights # list of weights, represented by a list containing weight values
        self.popsize = popsize # population size
        self.poffspring = poffspring # offspring probability
        self.pmut = pmut # mutation probability
        self.termiterations = termiterations # number of iterations until termination
        self.bestScoreList = []
        self.bestWeightsList = []

        # initialize population using weights and popsize
        # population is represented by list of weights
        # also create list containing fitness for each instance (compute once)
        self.population = []
        self.fitnesses = []
        for i in range(self.popsize):
            # STEP 1) append a random permutation of cities to population #TODO 1, how make initial population?
            tempweights = []
            for x in range(len(self.weights)):
                tempweights.append(np.random.random())
            self.population.append(tempweights)
            # STEP 2) evaluate quality candidate
            self.fitnesses.append(self.calculateFitness(self.population[i]))

        # add best result of initial population and the corresponding set of weights
        self.bestScoreList.append(max(self.fitnesses))
        self.bestWeightsList.append(self.population[self.fitnesses.index(max(self.fitnesses))])

    # Get score from weights
    def runTetris(self, weights = None):
        player = AI(weights)
        game = Game(player)
        game.new_game(pieceLimit = PIECELIMIT)
        return game.run_game()

    # calculate fitness of a specific instance of the population
    def calculateFitness(self, weights):
        #TODO 2, algorithm is going to play tetris and returns its score
        #fitness = sum(weights) #temp
        fitness = self.runTetris(tuple(weights))
        print("weights", weights, "gave a score of:", fitness)
        return fitness

    # evaluates quality of each candidate by updating the fitnesses list
    def evaluatePopulation(self):
        for i in range(self.popsize):
            self.fitnesses[i] = self.calculateFitness(i)

    # binary tournament selection
    # returns a single candidate (index)
    def binaryTournamentSelect(self):
        # randomly select two candidates
        candidate1 = random.randrange(self.popsize)
        candidate2 = random.randrange(self.popsize)

        # return fittest one (lowest total distance)
        if self.fitnesses[candidate1] > self.fitnesses[candidate2]:
            return candidate1
        else:
            return candidate2

    # return two new childs given two parents
    def generateOffspring(self, parent1, parent2):
        # initialize childs
        child1 = [-1] * len(self.weights)
        child2 = [-1] * len(self.weights)

        # get instances of parents from index
        parent1Instance = self.population[parent1]
        parent2Instance = self.population[parent2]

        # fill children
        for i in range (len(child1)):
            temprandom = np.random.random()
            if temprandom < 0.5:
                child1[i] = parent1Instance[i]
                child2[i] = parent2Instance[i]
            else:
                child2[i] = parent1Instance[i]
                child1[i] = parent2Instance[i]

        # return generated childs
        return child1, child2

    # apply mutation by swapping two random indices
    def doMutation(self, instance):
        # choose two random points
        instance[random.randrange(len(self.weights))] = np.random.random() #TODO random waarde die logisch is (mean, random waarde uit stddev, inverse, whatevs)
        return instance

    # TODO
    # STEP 3: run algorithm until termination condition satisfied
    def runEA(self):
        iterations = 0
        while iterations < self.termiterations:

            nextGeneration = [] # list containing the next generation
            # fill the next generation
            while len(nextGeneration) < self.popsize:

                # a: Select (two) candidate solutions for reproduction
                parent1 = self.binaryTournamentSelect()
                parent2 = self.binaryTournamentSelect()

                # b: Recombine selected candidates
                # apply probability
                child1 = []
                child2 = []
                if random.random() < self.poffspring:
                    child1, child2 = self.generateOffspring(parent1, parent2)
                else:
                    # if no crossover, set parents as offspring
                    child1 = self.population[parent1]
                    child2 = self.population[parent2]

                # c: Mutate the resulting candidates with probability
                if random.random() < self.pmut:
                    child1 = self.doMutation(child1)

                if random.random() < self.pmut:
                    child2 = self.doMutation(child2)

                # add to next generation intermediate list
                # unless list is already full (in case of odd population size number)
                nextGeneration.append(child1)
                if len(nextGeneration) < self.popsize:
                    nextGeneration.append(child2)

                
            # d: Evaluate the new candidates
            nextGenerationFitnesses = []
            for c in range(self.popsize):
                nextGenerationFitnesses.append(self.calculateFitness(nextGeneration[c]))

            # average fitness decrease. Not used, but can be interesting
            """
            averageFitnessDecrease = (sum(self.fitnesses) / len(self.fitnesses)) - (sum(nextGenerationFinesses) / len(nextGenerationFinesses))
            print("average path length difference: ", averageFitnessDecrease)
            """

            # e: Select candidates for the next generation
            # we select all of them, since we apply generational gap replacement
            self.population = nextGeneration
            self.fitnesses = nextGenerationFitnesses

            # every iteration, add best score of offspring to list
            self.bestScoreList.append(max(self.fitnesses))
            self.bestWeightsList.append(self.population[self.fitnesses.index(max(self.fitnesses))])

            # done with iteration
            iterations += 1

        # when done, return the list of best scores for each iteration
        return self.bestScoreList

bleh = SimpleEA([0.5, 0.5, 0.5, 0.5], 32, P_CROSSOVER, P_MUTATION, 10)
bleh.runEA()
#print(bleh.bestScoreList)
#print(bleh.bestWeightsList)

print(max(bleh.bestScoreList))
print(bleh.bestWeightsList[bleh.bestScoreList.index(max(bleh.bestScoreList))])