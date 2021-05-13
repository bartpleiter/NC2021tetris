#!/usr/bin/env python3

import argparse
from game import Game

from players import AI

PIECELIMIT = -1 # Maximum number of pieces in a game before game over. Set to -1 for unlimited

# Get score from weights
def runTetris(weights = None):
    player = AI(weights)
    game = Game(player)
    game.new_game(pieceLimit = PIECELIMIT)
    return game.run_game()


def geneticAlgorithm():
    # TODO: currently just runs a single game with some hardcoded weights
    weights = (0.5, -0.5, -0.5, -0.5, -0.5, -0.5)
    fitness = runTetris(weights)
    print("weights", weights, "gave a score of:", fitness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add something here if we want to add some command line args
    args = parser.parse_args()
    geneticAlgorithm()
