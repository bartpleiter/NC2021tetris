#!/usr/bin/env python3

import argparse
from game import Game

from players import AI


# Get score from weights
def runTetris(weights = None):
    player = AI(weights)
    game = Game(player)
    game.new_game()
    return game.run_game()


def geneticAlgorithm():
    # TODO: currently just runs a single game with some hardcoded weights
    weights = (-8, -18, -10.497, 16.432)
    fitness = runTetris(weights)
    print("weights", weights, "gave a score of:", fitness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add something here if we want to add some command line args
    args = parser.parse_args()
    geneticAlgorithm()
