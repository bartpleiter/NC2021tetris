import time
import curses
from game_board import NUM_COLUMNS, BORDER_WIDTH, BLOCK_WIDTH, PREVIEW_COLUMN

SHOW_AI = False
SHOW_AI_SPEED = 0.01
AI_DISPLAY_SCREEN = False

class Human(object):
    def __init__(self):
        self.name = 'Human'


class AI(object):

    def __init__(self, weights=None):
        self.weights = weights or (-8, -18, -10.497, 16.432)
        #self.totalMoves = 0

    def score_board(self, original_board, this_board):
        ### hier dat ding uitrekenen met heights ofzo (Jasper)
        ### en dan meegeven

        # OLD STUFF TODO DELETEETETETE
        #height_sum = get_height_sum(this_board)
        #holes = get_holes(this_board)
        #cumulative_holes = get_number_of_squares_above_holes(this_board)
        #score_diff = this_board.score - original_board.score

        A, B, C, D, E, F = self.weights
        score = (
            (A * fullRows) + # cleared lines (Jasper)
            (B * holes) + # number of holes (Bart)
            (C * holeDepth) + # cumulative hole depth (Bart)
            (D * bumpiness) + # the sum of height differences between adjacent columns (Jasper)
            (E * deepWells) + # sum of well depths of depth > 1 (Abel)
            (F * deltaHeight) # height difference between heighest and lowest (Abel)
        )
        print("Score of board:", score)
        exit(0)
        return score

    def get_moves(self, game_board, board_drawer):
        #self.totalMoves += 1
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

                    board._settle_shape(board.falling_shape)

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
        #print(end-start)

        return best_final_row_position, best_final_column_position, best_final_orientation


def get_holes(this_board):
    for yeet in this_board.array:
        print(yeet)
    #exit(0)
    hole_count = 0
    for row in this_board.array:
        for cell in row:
            if cell and not _cell_below_is_occupied(cell, this_board.array):
                hole_count += 1
    return hole_count


def get_number_of_squares_above_holes(this_board):
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


def _cell_below_is_occupied(cell, board):
    try:
        return board[cell.row_position + 1][cell.column_position]
    except IndexError:
        return True


def get_height_sum(this_board):
    return sum([20 - val.row_position for row in this_board.array for val in row if val])
