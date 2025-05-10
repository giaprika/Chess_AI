import os
import sys
import pygame
import pygame_menu
import chess
import chess.engine
import threading
import time
import torch

from pygame.locals import *
from menu_screen import MenuScreen

from mcts import MCTS
from model import AlphaZeroNet
from evalute_elo import model_get_best_move

# Constants for initial menu screen
MENU_WIDTH, MENU_HEIGHT = 540, 360
WIDTH, HEIGHT = MENU_WIDTH, MENU_HEIGHT
BOARD_SIZE = 540
SQ_SIZE = BOARD_SIZE // 8

WHITE = (237, 199, 126)
BLACK = (180, 104, 23)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game")
font = pygame.font.Font(None, 36)

# Load piece images
PIECE_IMAGE_MAP = {
    'P': 'pw', 'N': 'nw', 'B': 'bw', 'R': 'rw', 'Q': 'qw', 'K': 'kw',
    'p': 'pb', 'n': 'nb', 'b': 'bb', 'r': 'rb', 'q': 'qb', 'k': 'kb'
}
PIECE_IMAGES = {}
for piece, image_name in PIECE_IMAGE_MAP.items():
    image_path = f'data/images/{image_name}.png'
    try:
        PIECE_IMAGES[piece] = pygame.image.load(image_path)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        sys.exit()

def display_game_over(winner=None):
    result_text = "Game Over"
    if winner:
        result_text = f"{winner} wins!" if winner != "Draw" else "Draw"
    game_over_text = font.render(result_text, True, WHITE)
    screen.fill(BLACK)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))
    pygame.display.flip()
    pygame.time.wait(3000)

class ChessGame:
    def __init__(self):
        global WIDTH, HEIGHT, SQ_SIZE, BOARD_SIZE, screen
        WIDTH, HEIGHT = 540, 540
        BOARD_SIZE = min(WIDTH, HEIGHT)
        SQ_SIZE = BOARD_SIZE // 8
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.board = chess.Board()
        self.selected_square = None
        self.dragging_piece = None
        self.start_pos = None
        self.current_turn = chess.WHITE
        self.player_color = None
        self.model = None
        self.max_time = 2

        # Rescale piece images
        for key in PIECE_IMAGES:
            PIECE_IMAGES[key] = pygame.transform.scale(PIECE_IMAGES[key], (SQ_SIZE, SQ_SIZE))

    def draw_board(self):
        colors = [WHITE, BLACK]
        for row in range(8):
            for col in range(8):
                draw_row = 7 - row if self.player_color == chess.BLACK else row
                draw_col = 7 - col if self.player_color == chess.BLACK else col
                pygame.draw.rect(
                    screen, colors[(row + col) % 2],
                    pygame.Rect(draw_col * SQ_SIZE, draw_row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )

    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                board_row = 7 - row if self.player_color == chess.BLACK else row
                board_col = 7 - col if self.player_color == chess.BLACK else col
                piece = self.board.piece_at(chess.square(col, 7 - row))
                if piece:
                    if self.selected_square == chess.square(col, 7 - row) and self.dragging_piece:
                        continue
                    draw_x = board_col * SQ_SIZE
                    draw_y = board_row * SQ_SIZE
                    screen.blit(PIECE_IMAGES[piece.symbol()], (draw_x, draw_y))

    def draw_dragging_piece(self):
        if self.dragging_piece:
            pos = pygame.mouse.get_pos()
            screen.blit(self.dragging_piece, (pos[0] - SQ_SIZE // 2, pos[1] - SQ_SIZE // 2))

    def handle_events(self):
        if self.model is None or self.board.turn == self.player_color:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event)
                elif event.type == MOUSEBUTTONUP:
                    self.handle_mouse_up(event)
        else:
            if not hasattr(self, "ai_thread") or not self.ai_thread.is_alive():
                self.ai_thread = threading.Thread(target=self.run_engine_move)
                self.ai_thread.start()

    def run_engine_move(self):
        try:
            print("[AI] Thinking...")
            best_move = model_get_best_move(self.board)
            time.sleep(0.5)
            self.board.push(best_move)
            self.current_turn = not self.current_turn
            print("[AI] Move done.")
        except Exception as e:
            print(f"[ERROR] AI move failed: {e}")

    def handle_mouse_down(self, event):
        row, col = event.pos[1] // SQ_SIZE, event.pos[0] // SQ_SIZE
        if self.player_color == chess.BLACK:
            row, col = 7 - row, 7 - col
        square = chess.square(col, 7 - row)
        piece = self.board.piece_at(square)
        if piece and piece.color == self.current_turn:
            self.selected_square = square
            self.start_pos = (row, col)
            self.dragging_piece = PIECE_IMAGES[piece.symbol()]

    def handle_mouse_up(self, event):
        if not self.dragging_piece:
            return
        end_row, end_col = event.pos[1] // SQ_SIZE, event.pos[0] // SQ_SIZE
        if self.player_color == chess.BLACK:
            end_row, end_col = 7 - end_row, 7 - end_col
        end_square = chess.square(end_col, 7 - end_row)
        move = None
        for legal_move in self.board.legal_moves:
            if legal_move.from_square == self.selected_square and legal_move.to_square == end_square:
                move = legal_move
                break
        if move:
            self.board.push(move)
            self.current_turn = not self.current_turn
        self.selected_square = None
        self.dragging_piece = None

    def run(self):
        clock = pygame.time.Clock()
        while not self.board.is_game_over():
            self.handle_events()
            self.draw_board()
            self.draw_pieces()
            self.draw_dragging_piece()
            pygame.display.flip()
            clock.tick(60)
        result = self.board.result()
        winner = "Draw" if result == "1/2-1/2" else ("White" if result == "1-0" else "Black")
        display_game_over(winner)

# === Menu functions ===

def start_pvp():
    ChessGame().run()

def start_pvc():
    game = ChessGame()
    game.player_color = chess.WHITE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    game.model = model
    game.run()

if __name__ == '__main__':
    screen = pygame.display.set_mode((540, 360))
    menu = MenuScreen(screen, start_pvp, start_pvc)
    menu.main_loop()