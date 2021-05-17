#!/usr/bin/env python
# coding: utf-8

# Natural evolution strategies (NES) adapted from https://arxiv.org/abs/1703.03864 and implementation from https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d

# In[10]:


import numpy as np
import argparse
from game import Game
from players import AI


import concurrent.futures


# In[11]:


class NES:

  def __init__(self, weights, steps, sigma, learningrate, population, piecelimit):
    self.weights = weights
    self.steps = steps
    self.sigma = sigma
    self.learningrate = learningrate
    self.population = population
    self.piecelimit = piecelimit

  def runTetris(self, weights = None):
    player = AI(weights)
    game = Game(player)
    game.new_game(pieceLimit = self.piecelimit)
    reward = game.run_game()
    return reward

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
          
          #solution = self.weights + self.sigma*N[j]
          #R[j] = self.runTetris(tuple(solution))
          
        for t in processList:
         
          X.append(t.result())
          
       
      standardized_rewards = (X - np.mean(X)) / np.std(X)
      
      #standardized_rewards = (R - np.mean(R)) / np.std(R)
      grad = np.dot(N.T, standardized_rewards)/(population * sigma)
      self.weights += self.learningrate * grad


      if i % 20 == 0:
        print('iter %d. w: %s, reward: %f' % 
          (i, str(self.weights), self.runTetris(tuple(self.weights))))



# In[12]:


weights = [0.3, 0.4, 0.5, -0.3, -0.4, -0.5] #[-1, -1, -1, 1]
steps = 300
sigma = 0.1
learningrate = 0.01
population = 50
piecelimit = -1



samplerun = NES(weights, steps, sigma, learningrate, population, piecelimit)
samplerun.optimize()


# In[ ]:




