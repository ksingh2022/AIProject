# Karanjit Singh
# DCSI 6612: Intro to AI
# Final Project: Connect 4 with AI Agent

# Libraries to Import
import numpy as np
import random
import pygame
import sys
import math

# Colors used throughout the game
BLUE = (0, 0, 205)
BLACK = (0, 0, 0)
RED = (245, 0, 0)
YELLOW = (245, 245, 0)
Pink = (255, 105, 180)
# Columns and Rows for setup
Num_rows = 6
Num_col = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4


def make_board():
    PlayingBoard = np.zeros((Num_rows, Num_col))
    return PlayingBoard


def drop_piece(PlayingBoard, row, col, piece):
    PlayingBoard[row][col] = piece


def check_location(PlayingBoard, col):
    return PlayingBoard[Num_rows - 1][col] == 0


def get_next_open_row(PlayingBoard, col):
    for r in range(Num_rows):
        if PlayingBoard[r][col] == 0:
            return r


def print_PlayingBoard(PlayingBoard):
    print(np.flip(PlayingBoard, 0))


# Checking to see if player won after the move

def winning_move(PlayingBoard, piece):
    # Check horizontally
    for c in range(Num_col - 3):
        for r in range(Num_rows):
            if PlayingBoard[r][c] == piece and PlayingBoard[r][c + 1] == piece and PlayingBoard[r][c + 2] == piece and \
                    PlayingBoard[r][
                        c + 3] == piece:
                return True

    # Check vertically
    for c in range(Num_col):
        for r in range(Num_rows - 3):
            if PlayingBoard[r][c] == piece and PlayingBoard[r + 1][c] == piece and PlayingBoard[r + 2][c] == piece and \
                    PlayingBoard[r + 3][
                        c] == piece:
                return True
    # Checking Sloped Diagonals
    # Check for Pos
    for c in range(Num_col - 3):
        for r in range(Num_rows - 3):
            if PlayingBoard[r][c] == piece and PlayingBoard[r + 1][c + 1] == piece and PlayingBoard[r + 2][
                c + 2] == piece and PlayingBoard[r + 3][
                c + 3] == piece:
                return True

    # Check for neg
    for c in range(Num_col - 3):
        for r in range(3, Num_rows):
            if PlayingBoard[r][c] == piece and PlayingBoard[r - 1][c + 1] == piece and PlayingBoard[r - 2][
                c + 2] == piece and PlayingBoard[r - 3][
                c + 3] == piece:
                return True


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score


# checking moves made and scores, looking to see if 4 in row
def score_position(PlayingBoard, piece):
    score = 0

    # center column
    center_array = [int(i) for i in list(PlayingBoard[:, Num_col // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Horizontal
    for r in range(Num_rows):
        row_array = [int(i) for i in list(PlayingBoard[r, :])]
        for c in range(Num_col - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Vertical
    for c in range(Num_col):
        col_array = [int(i) for i in list(PlayingBoard[:, c])]
        for r in range(Num_rows - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # positive diagonal
    for r in range(Num_rows - 3):
        for c in range(Num_col - 3):
            window = [PlayingBoard[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)
    # Neg Diagonal
    for r in range(Num_rows - 3):
        for c in range(Num_col - 3):
            window = [PlayingBoard[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(PlayingBoard):
    return winning_move(PlayingBoard, PLAYER_PIECE) or winning_move(PlayingBoard, AI_PIECE) or len(
        get_valid_locations(PlayingBoard)) == 0


# Implementing the minimax part of project of AI agent
# Agent is looking for open positions and locations and verifying where it is best place to place piece
# using alpha beta algorithm for decisions

def minimax(PlayingBoard, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(PlayingBoard)
    is_terminal = is_terminal_node(PlayingBoard)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(PlayingBoard, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(PlayingBoard, PLAYER_PIECE):
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(PlayingBoard, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(PlayingBoard, col)
            b_copy = PlayingBoard.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(PlayingBoard, col)
            b_copy = PlayingBoard.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def get_valid_locations(PlayingBoard):
    valid_locations = []
    for col in range(Num_col):
        if check_location(PlayingBoard, col):
            valid_locations.append(col)
    return valid_locations


def Next_best_move(PlayingBoard, piece):
    valid_locations = get_valid_locations(PlayingBoard)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(PlayingBoard, col)
        temp_board = PlayingBoard.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def draw_board(PlayingBoard):
    for c in range(Num_col):
        for r in range(Num_rows):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(Num_col):
        for r in range(Num_rows):
            if PlayingBoard[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif PlayingBoard[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


PlayingBoard = make_board()
print_PlayingBoard(PlayingBoard)
game_over = False

pygame.init()

SQUARESIZE = 100

width = Num_col * SQUARESIZE
height = (Num_rows + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
draw_board(PlayingBoard)
pygame.display.update()

DisplayFont = pygame.font.SysFont("arial", 75)

turn = random.randint(PLAYER, AI)

while not game_over:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)

        pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            # print(event.pos)
            # Ask for Player 1 Input
            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))

                if check_location(PlayingBoard, col):
                    row = get_next_open_row(PlayingBoard, col)
                    drop_piece(PlayingBoard, row, col, PLAYER_PIECE)

                    if winning_move(PlayingBoard, PLAYER_PIECE):
                        label = DisplayFont.render("Player 1 wins!!", 1, Pink)
                        screen.blit(label, (40, 10))
                        game_over = True

                    turn += 1
                    turn = turn % 2

                    print_PlayingBoard(PlayingBoard)
                    draw_board(PlayingBoard)

    #  Ask for user input
    if turn == AI and not game_over:

        col, minimax_score = minimax(PlayingBoard, 5, -math.inf, math.inf, True)

        if check_location(PlayingBoard, col):

            row = get_next_open_row(PlayingBoard, col)
            drop_piece(PlayingBoard, row, col, AI_PIECE)

            if winning_move(PlayingBoard, AI_PIECE):
                label = DisplayFont.render("Player 2 wins!!", 1, Pink)
                screen.blit(label, (40, 10))
                game_over = True

            print_PlayingBoard(PlayingBoard)
            draw_board(PlayingBoard)

            turn += 1
            turn = turn % 2

    if game_over:
        pygame.time.wait(3000)
