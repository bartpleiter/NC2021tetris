#!/usr/bin/env python3


# coding: utf-8

# Natural evolution strategies (NES) adapted from https://arxiv.org/abs/1703.03864 and implementation from https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d

# In[10]:


import numpy as np
np.seterr(all='raise') #raise errors instead of printing a warning in order to catch them as an exception.
import argparse
import warnings


from game import Game
from players import AI

import concurrent.futures


# In[11]:


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

  def log_results(self, reward, iteration):

  	with open('NES_results/'+ str(self.run), 'a') as file:
  		file.write(str(iteration) + ',' + str(self.weights) + ',' + str(reward) + '\n')




  def optimize(self):
    for i in range(self.steps):

      
    
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

      except FloatingPointError: #Flawed solution to the problem, need to think about it
      	X = [x + 0.0001 for x in X]
      	standardized_rewards = (X - np.mean(X)) / np.std(X)
      	grad = np.dot(N.T, standardized_rewards)/(population * sigma)
      	print('Flawed solution', grad)
      
      #standardized_rewards = (R - np.mean(R)) / np.std(R)
      
      self.weights += self.learningrate * grad

      reward = self.runTetris(tuple(self.weights))
      if i % 1 == 0:
        print('iter %d. w: %s, reward: %d' % 
          (i, str(self.weights), reward))
      if self.log == True:
      	self.log_results(reward, i)






#Parameters
weights = [0.3, -0.4, -0.5, -0.3, -0.4, -0.5, -0.1, 0.4] #working weights
steps = 5 #Amount of iterations per run
sigma = 0.1 #size of gaussian sampling
learningrate = 0.01 #learningrate 
population = 50 #number of weights samples from the gaussian distribution
piecelimit = -1 #piecelimit for the game (-1 is unlimited)
runs = 5 #Number of runs
log = True #When set to True it will create a log file per run with results
experiment_name = 'test' #this name will be the name of your file + corresponding run number



#Multiple runs with random intialized weights TO DO: think about intialization
for run in range(runs):
	experiment = str(run) + '_' + experiment_name
	weights = np.random.uniform(low = -1.5, high = 1.5, size= (8,))
	samplerun = NES(weights, steps, sigma, learningrate, population, piecelimit, experiment, log)
	samplerun.optimize()


# In[ ]:



