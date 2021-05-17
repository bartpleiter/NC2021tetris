import time
import curses
from game_board import NUM_COLUMNS, NUM_ROWS, BORDER_WIDTH, BLOCK_WIDTH, PREVIEW_COLUMN

SHOW_AI = True
SHOW_AI_SPEED = 0.01
AI_DISPLAY_SCREEN = True
DEBUG_SCORE = False

##########################
# FEATURES/SCORE FUNCTIONS
##########################

# (Jasper)
# Calculates the number of full horizontal rows
def getFullRows(board):
    rows = 0
    for row in board.array:
        if all(row):
            rows += 1
    if DEBUG_SCORE:
        print("Rows:", rows)
    return rows

# (Bart)
# Calculates the number of holes in each column (so a hole that spans over two columns count as two).
# Holes of multiple cells deep count as a single hole
def getHoles(board):
    holes = 0
    for r, row in enumerate(board.array):
        if r > 0: # ceiling does not count!
            for c, cell in enumerate(row):
                if not cell and board.array[r-1][c]: # index always safe, because the top row is skipped
                    holes += 1
    if DEBUG_SCORE:
        print("Holes:", holes)
    return holes

# (Bart)
# Calculates the cumulative hole depth
# This is calculated by looking at the highest block, and counting the empty cells below (with a block above), 
#   each multiplied by the depth relative to this highest block
# Deep holes are only counted once (based on their top height, by only counting empty cells with a block above
def getHoleDepth(board):
    cumulativeHoleDepth = 0
    for c in range(NUM_COLUMNS): # for each column
        foundBlock = False
        blockHeight = 0 # used for calculating the hole depth
        for r in range(NUM_ROWS): # for each row
            if not foundBlock: # first we need to find the highest block
                if board.array[r][c]:
                    foundBlock = True
                    blockHeight = NUM_ROWS - r
            else: # if we found a block already, we are counting holes
                if not board.array[r][c]: # found a hole
                    if r > 0 and board.array[r-1][c]: # only count deep holes once, r>0 is just for safety although not needed
                        cumulativeHoleDepth += blockHeight - (NUM_ROWS - r)

    if DEBUG_SCORE:
        print("holeDepth:", cumulativeHoleDepth)
    return cumulativeHoleDepth

# (Jasper)
# Calculates the bumpiness of the field
# This is done by summing the height difference between each adjacent column, walls not included!
def getBumpiness(heights):
    bumpiness = 0
    for i in range(len(heights)-1):
        bumpiness += abs(heights[i] - heights[i+1])
    if DEBUG_SCORE:
        print("bumpiness:", bumpiness)
    return bumpiness

# (Abel)
# Calculates the cumulative well depth of wells deeper than 1
# A well is defined as follows:
#   A column has a well if the height of the left AND right adjacent column (walls included),
#       is larger than its own height. The well depth is then the SMALLEST height difference between
#       the column and its adjacent column
# A well can NOT be a hole, we only look at the height of a column
def getDeepWells(heights):
    cumulativeWellDepth = 0
    for i in range(len(heights)):
        if i == 0:
            wellDepth = heights[i+1] - heights[i]
            if wellDepth>1:
                cumulativeWellDepth += wellDepth
        elif i == (len(heights)-1):
            wellDepth = heights[len(heights)-2] - heights[len(heights)-1]
            if wellDepth>1:
                cumulativeWellDepth += wellDepth
        else:
            wellDepth = min(heights[i-1],heights[i+1]) - heights[i]
            if wellDepth>1:
                cumulativeWellDepth += wellDepth
    if DEBUG_SCORE:
        print("Deep wells:", cumulativeWellDepth)
    return cumulativeWellDepth

# (Abel)
# Calculates the difference between the highest and lowest column height
def getDeltaHeight(heights):
    deltaHeight = (max(heights)-min(heights))
    if DEBUG_SCORE:
        print("deltaHeight:", deltaHeight)
    return deltaHeight

# Calculates the heights of the columns on the board for easier and more efficient calculations
def getHeights(board):
    heights = [0] * board.num_columns
    for r in range (board.num_rows):
        for c in range (board.num_columns):
            if board.array[board.num_rows - 1 - r][c] is not None:
                heights[c] = r+1
    return heights

#################
# AI CODE
#################

class AI(object):

    def __init__(self, weights=None):
        self.weights = weights or (0.679, -0.944, -0.248, -0.265, -0.034, 0.037)

    def score_board(self, original_board, this_board):
        heights = getHeights(this_board)
        if DEBUG_SCORE:
            this_board.printSelf()

        fullRows = getFullRows(this_board) # cleared lines 
        holes = getHoles(this_board) #  (Bart)
        holeDepth = getHoleDepth(this_board) # cumulative hole depth (Bart)
        bumpiness = getBumpiness(heights) # the sum of height differences between adjacent columns (Jasper)
        deepWells = getDeepWells(heights) # sum of well depths of depth > 1 (Abel)
        deltaHeight = getDeltaHeight(heights) # height difference between heighest and lowest (Abel)


        A, B, C, D, E, F = self.weights
        score = (
            (A * fullRows) +
            (B * holes) +
            (C * holeDepth) +
            (D * bumpiness) +
            (E * deepWells) +
            (F * deltaHeight)
        )
        if DEBUG_SCORE:
            print("Score of board:", score)
            print()
        return score

    def get_moves(self, game_board, board_drawer):
        #start = time.time()
        max_score = -100000

        best_final_column_position = None
        best_final_row_position = None
        best_final_orientation = None

        falling_orientations = game_board.falling_shape.number_of_orientations
        next_orientations = game_board.next_shape.number_of_orientations

        originalBoard = game_board.deepBoardCopy()
        for column_position in range(-2, NUM_COLUMNS + 2): # we add -2 and +2 here to make sure all positions at all orientations are included
            for orientation in range(falling_orientations):
                board = originalBoard.deepBoardCopy()
                board.falling_shape.orientation = orientation
                board.falling_shape.move_to(column_position, 2)

                while not board.shape_cannot_be_placed(board.falling_shape):
                    board.falling_shape.lower_shape_by_one_row()
                board.falling_shape.raise_shape_by_one_row()
                if not board.shape_cannot_be_placed(board.falling_shape):
                    # now we have a valid possible placement
                    
                    # show placement of the AI
                    if SHOW_AI:
                        board_drawer.update_settled_pieces(board)  # clears out the old shadow locations
                        board_drawer.update_falling_piece(game_board)
                        board_drawer.update_shadow(board)
                        board_drawer.refresh_screen()

                    board._settle_shape_no_clear(board.falling_shape)

                    score = self.score_board(game_board, board)
                    if score > max_score:
                        max_score = score
                        best_final_column_position = board.falling_shape.column_position
                        best_final_row_position = board.falling_shape.row_position
                        best_final_orientation = board.falling_shape.orientation

                    if SHOW_AI:
                        board_drawer.stdscr.addstr(
                            BORDER_WIDTH + 14,
                            PREVIEW_COLUMN*BLOCK_WIDTH-2+BORDER_WIDTH,
                            'PLACEMENT SCORE: %d' % score,
                            curses.color_pair(7)
                        )
                        time.sleep(SHOW_AI_SPEED)

        #end = time.time()
        #print("Time used to find a placement:", end-start)

        return best_final_row_position, best_final_column_position, best_final_orientation


# OLD STUFF, can be deleted when other features are implemented
def old_get_holes(this_board):
    hole_count = 0
    for row in this_board.array:
        for cell in row:
            if cell and not _cell_below_is_occupied(cell, this_board.array):
                hole_count += 1
    return hole_count


def old_get_number_of_squares_above_holes(this_board):
    count = 0
    for column in range(this_board.num_columns):
        saw_hole = False
        for row in range(this_board.num_rows-1, 0, -1):
            cell = this_board.array[row][column]
            if cell:
                if saw_hole:
                    count += 1
            else:
                saw_hole = True
    return count

def old_get_height_sum(this_board):
    return sum([20 - val.row_position for row in this_board.array for val in row if val])

class Human(object):
    def __init__(self):
        self.name = 'Human'
