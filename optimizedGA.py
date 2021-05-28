#!/usr/bin/env python3


# Imports
import sys
import numpy as np
import math
import random
import concurrent.futures
import matplotlib.pyplot as plt
from game import Game
import time

from players import AI

PIECELIMIT = -1 # Maximum number of pieces in a game before game over. Set to -1 for unlimited

########################
# HYPER PARAMETERS
########################
P_MUTATION = 0.1 # mutation probability
P_CROSSOVER = 0.5 # crossover probability
P_POPULATIONSIZE = 100 # must be even
P_GENERATIONS = 32 
P_MUTATIONREDUCTION = True
P_BESTAMOUNT = 5
P_GOODAMOUNT = 25
LOG = True #When set to True it will create a log file per run with results
EXP_NAME = 'test Optimized'

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
    def __init__(self, weights, popsize = 100, poffspring = 0.5, pmut = 0.1, termgeneration = 10, reduceMutationRate = True, numberOfBest = 5, numberOfGood = 25, log = False, experiment_name = " ", run = 0):
        self.weights = weights # list of weights, represented by a list containing weight values
        self.popsize = popsize # population size
        self.poffspring = poffspring # offspring probability
        self.pmut = pmut # mutation probability
        self.termgeneration = termgeneration # number of generation until termination
        self.reduceMutationRate = reduceMutationRate # Whether or not the algorithm will use reducing mutation rate.
        self.numberOfBest = numberOfBest # Number of the best of population taken to next generation
        self.numberOfGood = numberOfGood # Number of the best + other good of population taken to next generation
        self.log = log #When set to True it will create a log file per run with results
        self.experiment_name = experiment_name #name for logging purposes
        self.run = run # run number for logging purposes
        self.bestScoreList = []
        self.bestWeightsList = []

        # initialize population using weights and popsize
        # population is represented by list of weights
        # also create list containing fitness for each instance (compute once)
        self.population = []
        for i in range(self.popsize):
            # STEP 1) append a random permutation of cities to population #TODO 1, how make initial population?
            tempweights = []
            for x in range(len(self.weights)):
                tempweights.append(np.random.uniform(-1, 1))
            self.population.append(self.normalize(tempweights))
        
        print("Running generation: 0");
        #Step 2) evaluate quality candidate
        self.fitnesses = []
        processList = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(self.popsize):
                processList.append(executor.submit(self.calculateFitness, self.population[i]))
                
            for t in processList:
                self.fitnesses.append(t.result())

        # add best result of initial population and the corresponding set of weights
        self.bestScoreList.append(max(self.fitnesses))
        self.bestWeightsList.append(self.population[self.fitnesses.index(max(self.fitnesses))])

        self.printGeneration(0)

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
        #print("weights", weights, "gave a score of:", fitness)
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

    def normalize(self, instance):
        tempabs = []
        for i in range(len(instance)):
            tempabs.append(abs(instance[i]))
        factor = tempabs[tempabs.index(max(tempabs))]
        for i in range(len(instance)):
            instance[i] = instance[i]/factor
        return instance

    # apply mutation by swapping two random indices
    def doMutation(self, instance):
        index = random.randrange(len(self.weights))
        instance[index] = instance[index] * np.random.uniform(-1.5, 1.5)
        return self.normalize(instance)

    def averageWeights(self):
        averageweights = []
        for i in range(len(self.population[0])):
           averageweights.append(0)
        for w in self.population:
            for i in range(len(w)):
                averageweights[i] += w[i]
        for i in range(len(averageweights)):
            averageweights[i] = averageweights[i]/len(self.population)
        return averageweights

    def printGeneration(self, generation):
        # Print generation results
        print("Best score for generation", generation, ":", max(self.fitnesses))
        print(" using weights:", [ '%.3f' % w for w in self.population[self.fitnesses.index(max(self.fitnesses))] ])
        print("Average score for generation", generation, ":", int(np.mean(self.fitnesses)))
        print("Average weights for generation", generation, ":",  [ '%.3f' % w for w in self.averageWeights() ])
        print("-------------------------")

    def log_results(self, generation):
        with open('OEA_results/'+ str(self.run), 'a') as file:
            toLog = (str(generation) + '|' 
                + ", ".join(['%.4f' % x for x in self.population[self.fitnesses.index(max(self.fitnesses))]]) 
                + '|' + str(max(self.fitnesses))
                + '|' + ", ".join(['%.4f' % x for x in self.averageWeights()]) 
                + '|' + str((sum(self.fitnesses)/len(self.fitnesses))))
            file.write(toLog + '\n')
        
    def eliteSelection(self, best, good):
        returnList = []

        sortedFitnesses = sorted(self.fitnesses, reverse=True)
        while len(returnList) < best:
            returnList.append(self.population[self.fitnesses.index(sortedFitnesses[len(returnList)])])

        while len(returnList) < good:
            returnList.append(self.population[self.binaryTournamentSelect()])

        return returnList

    # TODO
    # STEP 3: run algorithm until termination condition satisfied
    def runEA(self):
        generation = 0
        while generation < self.termgeneration:

            # list containing the next generation, prefilled with winners of binary tournament
            nextGeneration = self.eliteSelection(self.numberOfBest, self.numberOfGood)

            # fill the rest of the next generation
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
                if(self.reduceMutationRate):
                    if random.random() < (self.pmut * (self.termgeneration-generation)/self.termgeneration):
                        child1 = self.doMutation(child1)
                    if random.random() < (self.pmut * (self.termgeneration-generation)/self.termgeneration):
                        child2 = self.doMutation(child2)
                else:
                    if random.random() < self.pmut:
                        child1 = self.doMutation(child1)
                    if random.random() < self.pmut:
                        child2 = self.doMutation(child2)

                # add to next generation intermediate list
                # unless list is already full (in case of odd population size number)
                nextGeneration.append(child1)
                if len(nextGeneration) < self.popsize:
                    nextGeneration.append(child2)

            generation += 1
            print("Running generation:", generation);
                
            # d: Evaluate the new candidates
            nextGenerationFitnesses = []
            processList = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for c in range(self.popsize):
                    processList.append(executor.submit(self.calculateFitness, nextGeneration[c]))
                for t in processList:
                    nextGenerationFitnesses.append(t.result())

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

            self.printGeneration(generation)

            if self.log == True:
                self.log_results(generation)
            # done with iteration

        # when done, return the list of best scores for each iteration
        return self.bestScoreList

    
runs = 10 #Number of runs
for run in range(runs):
    start = time.time()
    experiment = str(run) + '_' + "Base"
    bleh = SimpleEA([None] * 8, P_POPULATIONSIZE, P_CROSSOVER, P_MUTATION, P_GENERATIONS, P_MUTATIONREDUCTION, P_BESTAMOUNT, P_GOODAMOUNT, LOG, experiment, run)
    bleh.runEA()
    end = time.time()
    with open('OEA_results/'+ "Base" + "_times", 'a') as file:
        file.write("Running time: " + str(end-start) + " seconds" + '\n')

for run in range(runs):
    start = time.time()
    experiment = str(run) + '_' + "LinMut"
    bleh = SimpleEA([None] * 8, P_POPULATIONSIZE, P_CROSSOVER, P_MUTATION, P_GENERATIONS, False, P_BESTAMOUNT, P_GOODAMOUNT, LOG, experiment, run)
    bleh.runEA()
    end = time.time()
    with open('OEA_results/'+ "LinMut" + "_times", 'a') as file:
        file.write("Running time: " + str(end-start) + " seconds" + '\n')

for run in range(runs):
    start = time.time()
    experiment = str(run) + '_' + "KeepBestHalf"
    bleh = SimpleEA([None] * 8, P_POPULATIONSIZE, P_CROSSOVER, P_MUTATION, P_GENERATIONS, P_MUTATIONREDUCTION, 50, 50, LOG, experiment, run)
    bleh.runEA()
    end = time.time()
    with open('OEA_results/'+ "KeepBestHalf" + "_times", 'a') as file:
        file.write("Running time: " + str(end-start) + " seconds" + '\n')

for es in range(3):
    for run in range(runs):
        start = time.time()
        experiment = str(run) + '_' + "EliteSelection_" + str(es+1)
        bleh = SimpleEA([None] * 8, P_POPULATIONSIZE, P_CROSSOVER, P_MUTATION, P_GENERATIONS, P_MUTATIONREDUCTION, (es+1)*10, (es+1)*20, LOG, experiment, run)
        bleh.runEA()
        end = time.time()
        with open('OEA_results/'+ "EliteSelection_" + str(es+1) + "_times", 'a') as file:
            file.write("Running time: " + str(end-start) + " seconds" + '\n')



#print("Final results:")
#print("Max score:", max(bleh.bestScoreList))
#print("Weights to get best score:", [ '%.4f' % w for w in bleh.bestWeightsList[bleh.bestScoreList.index(max(bleh.bestScoreList))] ])
