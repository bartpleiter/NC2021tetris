#!/usr/bin/env python3

# Parse log files from Optimized EA, assuming the format is correct

import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join


# get all logs from folder
allLogFiles = [f for f in listdir("OEA_results") if isfile(join("OEA_results", f))]
allLogFiles.sort()


# parse log files into lists
def parseLog(file):
    unparsedLog = []
    with open("OEA_results/" + file, 'r') as f:
        unparsedLog = [l.strip() for l in f.readlines()]

    parsedLog = []
    for log in unparsedLog:
        values = []
        values.append(int(log.split('|')[0])) # generation

        weights = log.split('|')[1] # weights
        weightList = [float(w) for w in weights.split(',')]
        values.append(weightList)

        values.append(int(log.split('|')[2])) # score

        weights = log.split('|')[3] # avg weights
        weightList = [float(w) for w in weights.split(',')]
        values.append(weightList)

        values.append(float(log.split('|')[4])) # avg score

        parsedLog.append(values)

    return parsedLog

parsedLogs = [parseLog(f) for f in allLogFiles]
# parsedLogs[run][generation][thing]



GENERATIONS = list(range(len(parsedLogs[0])))
RUNS = len(parsedLogs)
IDX_GEN = 0
IDX_MAXWEIGHT = 1
IDX_MAXSCORE = 2
IDX_AVGWEIGHT = 3
IDX_AVGSCORE = 4
WEIGHTNAMES = ["FullRows", "Holes", "HoleDepth", "Bumpiness", "DeepWells", "DeltaHeight", "ShallowWells", "PatternDiversity"]



# plot weights

meanWeightList = []
stdWeightList = []

for gen in GENERATIONS:
    weights = [x[IDX_AVGWEIGHT] for x in [run[gen] for run in parsedLogs]]

    meanWeights = []
    stdWeights = []
    for w in range(len(weights[0])):
        a = [run[w] for run in weights]
        meanWeights.append(np.mean(a))
        stdWeights.append(np.std(a))
    meanWeightList.append(meanWeights)
    stdWeightList.append(stdWeights)

for w in range(len(meanWeightList[0])):
    plt.errorbar(GENERATIONS, [x[w] for x in meanWeightList], yerr=[x[w] for x in stdWeightList], label=WEIGHTNAMES[w]) 
    plt.xlabel("Generation")
    plt.ylabel("Weight value")
    plt.title("Avg weights for each generation, with std error over " + str(RUNS) + " runs")

plt.legend()
plt.show()


# plot weights

meanWeightList = []
stdWeightList = []

for gen in GENERATIONS:
    weights = [x[IDX_MAXWEIGHT] for x in [run[gen] for run in parsedLogs]]

    meanWeights = []
    stdWeights = []
    for w in range(len(weights[0])):
        a = [run[w] for run in weights]
        meanWeights.append(np.mean(a))
        stdWeights.append(np.std(a))
    meanWeightList.append(meanWeights)
    stdWeightList.append(stdWeights)

for w in range(len(meanWeightList[0])):
    plt.errorbar(GENERATIONS, [x[w] for x in meanWeightList], yerr=[x[w] for x in stdWeightList], label=WEIGHTNAMES[w]) 
    plt.xlabel("Generation")
    plt.ylabel("Weight value")
    plt.title("Max weights for each generation, with std error over " + str(RUNS) + " runs")

plt.legend()
plt.show()


# plot scores
meanScores = []
stdScores = []

for gen in GENERATIONS:
    scores = [x[IDX_MAXSCORE] for x in [run[gen] for run in parsedLogs]]
    meanScores.append(np.mean(scores))
    stdScores.append(np.std(scores))

plt.errorbar(GENERATIONS, meanScores, yerr=stdScores) 
plt.xlabel("Generation")
plt.ylabel("Weight")
plt.title("Max score for each generation, with std error over " + str(RUNS) + " runs")
plt.show()





# plot avg scores
meanScores = []
stdScores = []

for gen in GENERATIONS:
    scores = [x[IDX_AVGSCORE] for x in [run[gen] for run in parsedLogs]]
    meanScores.append(np.mean(scores))
    stdScores.append(np.std(scores))

plt.errorbar(GENERATIONS, meanScores, yerr=stdScores) 
plt.xlabel("Generation")
plt.ylabel("Weight")
plt.title("Average score for each generation, with std error over " + str(RUNS) + " runs")
plt.show()







"""
# plot weights
weightNames = ["FullRows", "Holes", "HoleDepth", "Bumpiness", "DeepWells", "DeltaHeight", "ShallowWells", "PatternDiversity"]
for i in range(len(weightNames)):

    for idx, log in enumerate(parsedLogs):
        gen = [l[0] for l in log]
        weight = [l[1][i] for l in log]
        plt.plot(gen, weight, label = "run " + str(idx + 1))

plt.xlabel("Generation")
plt.ylabel("Weight")
plt.title(weightNames[i] + " weight per generation for each run")
plt.legend()
plt.show()
"""