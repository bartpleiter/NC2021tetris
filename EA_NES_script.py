#!/usr/bin/env python3


# coding: utf-8

# Natural evolution strategies (NES) adapted from https://arxiv.org/abs/1703.03864 and implementation from https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d

# In[10]:


import numpy as np
np.seterr(all='raise') #raise errors instead of printing a warning in order to catch them as an exception.
import argparse
import warnings
import time

from game import Game
from players import AI

import concurrent.futures


class NES:

  def __init__(self, weights, steps, sigma, learningrate, population, piecelimit, run, log):
    self.weights = weights
    self.steps = steps
    self.sigma = sigma
    self.learningrate = learningrate
    self.population = population
    self.piecelimit = piecelimit
    self.run = run
    self.log = log

  def runTetris(self, weights = None):
    player = AI(weights)
    game = Game(player)
    game.new_game(pieceLimit = self.piecelimit)
    reward = game.run_game()
    return reward

  def log_results(self, reward, iteration, failed, failed_weights):

    with open('NES_results/'+ str(self.run), 'a') as file:
      toLog = (str(iteration) + '|' + ", ".join(["{:.4f}".format(w) for w in self.weights]) + '|' + str(reward))
      file.write(toLog + '\n')

    if failed:
      with open('NES_results/'+ str(self.run)+'failed', 'a') as file:
        toLog = (str(iteration) + '|' + ", ".join(["{:.4f}".format(w) for w in failed_weights]) + '|' + str(reward))
        file.write(toLog + '\n')


  def optimize(self):
    fail_counter = 0
    i = 0
    failed = False

    while i < self.steps:
      print("iteration :", i)
      failed_weights = self.weights

    
      N = np.random.randn(self.population, len(self.weights))
      R = np.zeros(self.population)
      X = []
      solutions = []
      processList = []
     
      for j in range(self.population):
          solutions.append(self.weights + self.sigma*N[j])

      with concurrent.futures.ProcessPoolExecutor() as executor:
        for j in range(self.population):
          processList.append(executor.submit(self.runTetris, tuple(solutions[j])))
        
        for t in processList:         
          X.append(t.result())
      
      #Try and catch for calculating the gradient in case of rewards being 0.
      try:
        standardized_rewards = (X - np.mean(X)) / np.std(X)
        grad = np.dot(N.T, standardized_rewards)/(population * sigma)
        self.weights += self.learningrate * grad

      except FloatingPointError:
        self.weights = np.random.uniform(low = -1.5, high = 1.5, size= (8,))
        fail_counter += 1 
        i = 0
        failed = True
        with open('NES_results/'+ str(self.run), 'w') as file:
          file.truncate()
          file.close()



      
      #standardized_rewards = (R - np.mean(R)) / np.std(R)
      
      
 
      reward = self.runTetris(tuple(self.weights))
      if i % 1 == 0:
        print('iter %d. w: %s, reward: %d' % 
          (i, str(self.weights), reward))

      if self.log == True and failed == False:
        self.log_results(reward, i, False, 0)
      elif self.log == True and failed == True:
        self.log_results(reward, i,True, failed_weights)
      failed = False
      i += 1





def run_experiment(steps, sigma, learningrate, population, piecelimit,  runs, log,  experiment_name):
  for run in range(runs):
    start = time.time()
    experiment = str(run) + '_' + experiment_name
    weights = np.random.uniform(low = -1.5, high = 1.5, size= (8,))
    samplerun = NES(weights, steps, sigma, learningrate, population, piecelimit, experiment, log)
    samplerun.optimize()
    end = time.time()
    runtime = end - start
    with open('NES_results/'+ 'runtime', 'a') as file:
          toLog = str(run) + '_' + (str(experiment) + '|' + str(runtime))
          file.write(toLog + '\n')

weights = [0.3, -0.4, -0.5, -0.3, -0.4, -0.5, -0.1, 0.4] #working weights
"""
---------------------------------
Baseline settings for experiments
---------------------------------
"""
steps = 32#Amount of iterations per run
sigma = 0.1 #size of gaussian sampling
learningrate = 0.01 #learningrate 
population = 100 #number of weights samples from the gaussian distribution
piecelimit = -1 #piecelimit for the game (-1 is unlimited)
runs = 10#Number of runs
log = True #When set to True it will create a log file per run with results
experiment_name = 'baseline_32_01_001_100_-1_10' #this name will be the name of your file + corresponding run number


"""
------------------------
Baseline experimenst
------------------------
"""
"""
run_experiment(steps,sigma, learningrate, population,piecelimit,runs, log, experiment_name)
run_experiment(32, 0.1, 0.01, 50, -1, 10, True, 'baseline_red_pop_32_01_001_50_-1_10')
run_experiment(32, 0.1, 0.01, 25, -1, 10, True, 'baseline_red_pop2_32_01_001_25_-1_10')
run_experiment(64, 0.1, 0.01, 100, -1, 10, True, 'baseline_increasedsteps_64_01_001_100_-1_10')
run_experiment(32, 0.2, 0.01, 100, -1, 10, True, 'baseline_increased_sigma_32_02_001_50_-1_10')
run_experiment(32, 0.1, 0.05, 100, -1, 10, True, 'baseline_increased_lr_32_01_001_50_-1_10')
run_experiment(32, 0.1, 0.005, 100, -1, 10, True, 'baseline_reduced_lr_32_01_001_50_-1_10')
run_experiment(100, 0.1, 0.01, 100, -1, 10, True, 'beefedup_100_01_001_100_-1_10')
run_experiment(100, 0.1, 0.01, 50, -1, 10, True, 'beefedup_red_pop_100_01_001_50_-1_10')
run_experiment(100, 0.1, 0.01, 25, -1, 10, True, 'beefedup_red_pop2_100_01_001_25_-1_10')
run_experiment(100, 0.1, 0.05, 100, -1, 10, True, 'beefedup_increased_lr_32_01_005_100_-1_10')
run_experiment(100, 0.1, 0.005, 100, -1, 10, True, 'beefedup_reduced_lr_32_01_0005_100_-1_10')
"""

"""
-----------------------
Optimizing runs
-----------------------
"""

#run_experiment(32, 0.1, 0.01, 100, -1, 10, True, 'baseline_increased_lr2_32_01_002_100_-1_10')
#run_experiment(32, 0.1, 0.015, 100, -1, 10, True, 'baseline_increased_lr3_32_01_0015_100_-1_10')
run_experiment(100, 0.1, 0.015, 100, -1, 10, True, 'beefedup_increased_lr2_32_01_0015_100_-1_10')
run_experiment(100, 0.1, 0.02, 100, -1, 10, True, 'beefedup_increased_lr3_32_01_002_100_-1_10')




