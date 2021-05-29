#!/usr/bin/env python3

# Parse log files from Optimized EA, assuming the format is correct

import numpy as np
import matplotlib.pyplot as plt
import sys
from os import listdir
from os.path import isfile, join
import scipy.stats as st

if len(sys.argv) < 3:
    print("Please give a log filter argument (specifying which logs to read), and an EXP number for in the plot titles")
    exit()

EXP_NR = sys.argv[2]

# get all logs from folder
allLogFiles = [f for f in listdir("OEA_results") if isfile(join("OEA_results", f)) and sys.argv[1] in f and "_times" not in f]
allLogFiles.sort()

print("Found", len(allLogFiles), "logs")

if len(allLogFiles) == 0:
    print("Found no matching logs")
    exit()


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



CONFIDENCE_INTERVAL = 0.95
GENERATIONS = list(range(len(parsedLogs[0])))
RUNS = len(parsedLogs)
IDX_GEN = 0
IDX_MAXWEIGHT = 1
IDX_MAXSCORE = 2
IDX_AVGWEIGHT = 3
IDX_AVGSCORE = 4
WEIGHTNAMES = ["FullRows", "Holes", "HoleDepth", "Bumpiness", "DeepWells", "DeltaHeight", "ShallowWells", "PatternDiversity"]
COLORS = ["blue", "orange", "green", "red", "purple", "brown", "violet", "grey"]


# plot weights

meanWeightList = []
stdWeightList = []
plt.figure()

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

for i, w in enumerate(range(len(meanWeightList[0]))):
    plt.plot(GENERATIONS, [x[w] for x in meanWeightList], label=WEIGHTNAMES[w], color=COLORS[i]) 
    mean = np.array([x[w] for x in meanWeightList])
    std = np.array([x[w] for x in stdWeightList])
    plt.fill_between(GENERATIONS, mean-std, mean+std, color=COLORS[i], alpha=.05)
    plt.xlabel("Generation")
    plt.ylabel("Weight value")
    plt.title("EXP "+EXP_NR+"\nWeights of average individual, averaged over " + str(RUNS) + " runs, with STD error")

plt.legend()
plt.savefig("Plots/Optimized_EXP "+EXP_NR+ "_"+ sys.argv[1] +"_avgWeights.png")
#plt.show()
plt.figure()


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

for i, w in enumerate(range(len(meanWeightList[0]))):
    plt.plot(GENERATIONS, [x[w] for x in meanWeightList], label=WEIGHTNAMES[w], color=COLORS[i]) 
    mean = np.array([x[w] for x in meanWeightList])
    std = np.array([x[w] for x in stdWeightList])
    plt.fill_between(GENERATIONS, mean-std, mean+std, color=COLORS[i], alpha=.05)
    plt.xlabel("Generation")
    plt.ylabel("Weight value")
    plt.title("EXP "+EXP_NR+"\nWeights of best individual, averaged over " + str(RUNS) + " runs, with STD error")

plt.legend()
plt.savefig("Plots/Optimized_EXP "+EXP_NR+ "_"+ sys.argv[1] +"_bestWeights.png")
#plt.show()
plt.figure()

# plot scores
meanScores = []
stdScores = []
rangeScores = []

for gen in GENERATIONS:
    scores = [x[IDX_MAXSCORE] for x in [run[gen] for run in parsedLogs]]
    meanScores.append(np.mean(scores))
    stdScores.append(np.std(scores))
    rangeScores.append((np.min(scores), np.max(scores)))
    

lowestScores = [c[0] for c in rangeScores]
highestScores = [c[1] for c in rangeScores]

stdScores = np.array(stdScores)
meanScores = np.array(meanScores)

plt.plot(GENERATIONS, meanScores) 
plt.fill_between(GENERATIONS, meanScores, meanScores + stdScores, color='r', alpha=.1) 
plt.fill_between(GENERATIONS, lowestScores, meanScores, color='b', alpha=.1) 
plt.xlabel("Generation")
plt.ylabel("Score")
plt.title("EXP "+EXP_NR+": Max score for each generation averaged over " + str(RUNS) + " runs, \nthe upper STD in red, and the lower bound in blue")
plt.savefig("Plots/Optimized_EXP "+EXP_NR+ "_"+ sys.argv[1] +"_maxScore.png")
#plt.show()
plt.figure()



# plot scores
meanScores = []
stdScores = []
rangeScores = []

for gen in GENERATIONS:
    scores = [x[IDX_AVGSCORE] for x in [run[gen] for run in parsedLogs]]
    meanScores.append(np.mean(scores))
    stdScores.append(np.std(scores))
    rangeScores.append((np.min(scores), np.max(scores)))
    

lowestScores = [c[0] for c in rangeScores]
highestScores = [c[1] for c in rangeScores]

stdScores = np.array(stdScores)
meanScores = np.array(meanScores)

plt.plot(GENERATIONS, meanScores) 
plt.fill_between(GENERATIONS, meanScores, meanScores + stdScores, color='r', alpha=.1) 
plt.fill_between(GENERATIONS, lowestScores, meanScores, color='b', alpha=.1) 
plt.xlabel("Generation")
plt.ylabel("Score")
plt.title("EXP "+EXP_NR+": Mean score for each generation averaged over " + str(RUNS) + " runs, \nthe upper STD in red, and the lower bound in blue")
plt.savefig("Plots/Optimized_EXP "+EXP_NR+ "_"+ sys.argv[1] +"_meanScore.png")
#plt.show()
