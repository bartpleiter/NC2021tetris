#!/usr/bin/env python3

# Parse log files from EA, assuming the format is correct

import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join


# get all logs from folder
allLogFiles = [f for f in listdir("NES_results") if isfile(join("NES_results", f))]
allLogFiles.sort()


# parse log files into lists
def parseLog(file):
    unparsedLog = []
    with open("NES_results/" + file, 'r') as f:
        unparsedLog = [l.strip() for l in f.readlines()]

    parsedLog = []
    for log in unparsedLog:
        values = []
        values.append(int(log.split('|')[0])) # iteration

        weights = log.split('|')[1] # weights
        weightList = [float(w) for w in weights.split(',')]
        values.append(weightList)

        values.append(int(log.split('|')[2])) # score
        parsedLog.append(values)

    return parsedLog

parsedLogs = [parseLog(f) for f in allLogFiles]


# plot scores
gen = [l[0] for l in parsedLogs[0]]
score = [l[2] for l in parsedLogs[0]]

for idx, log in enumerate(parsedLogs):
    gen = [l[0] for l in log]
    score = [l[2] for l in log]
    plt.plot(gen, score, label = "run " + str(idx + 1))

plt.xlabel("Generation")
plt.ylabel("Score")
plt.title("Score per generation for each run")
plt.legend()
plt.show()



# plot weights
weightNames = ["FullRows", "Holes", "HoleDepth", "Bumpiness", "DeepWells", "DeltaHeight", "ShallowWells", "PatternDiversity"]
for i in range(len(weightNames)):

    for idx, log in enumerate(parsedLogs):
        gen = [l[0] for l in log]
        weight = [l[1][i] for l in log]
        plt.plot(gen, weight, label = "run " + str(idx + 1))

    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(weightNames[i] + " weight per generation for each run")
    plt.legend()
    plt.show()
