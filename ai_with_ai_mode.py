import pygame
import torch
from model import AlphaZeroNet
from utils import board_to_tensor
from game_init import GameInit
from config import *
from mcts import MCTS
from evalute_elo import model_get_best_move

class AIWithAIMode(GameInit):
    def __init__(self):
        super().__init__()
        self.screen = pygame.display.set_mode((WIDTH_WINDOW_AI, HEIGHT_WINDOW_AI))
        self.manager.set_window_resolution((WIDTH_WINDOW_AI, HEIGHT_WINDOW_AI))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet().to(self.device)
        self.model.load_state_dict(torch.load("./data/model.pt", map_location=self.device))
        self.model.eval()

        self.mcts = MCTS(self.model, 2.0)

        self.ai_turn = True
        self.moveMade = False
        self.signal = True
        self.gameOver = False
        self.clock = pygame.time.Clock()

    def mainLoop(self):
        while self.running:
            self.time_delta = self.clock.tick(MAX_FPS) / 1000
            self.__eventHandler()

            if not self.gs.board.is_game_over():
                self.__AIMove()
                self.ai_turn = not self.ai_turn  # 🔄 Đổi lượt AI

            self.manager.update(self.time_delta)
            self.drawGameScreen()
            pygame.display.update()

    def __AIMove(self):
        board = self.gs.board
        best_move = model_get_best_move(board)

        if best_move:
            captured_piece = board.piece_at(best_move.to_square)
            self.gs.makeMove(best_move)
            if captured_piece:
                pygame.mixer.Sound.play(self.sound_capture)
            else:
                pygame.mixer.Sound.play(self.sound_move)
            self.moveMade = True

    def __eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game Quit")
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.__reset()
                elif event.key == pygame.K_z:
                    self.gs.undoMove()
                    self.editAIPanel()
                    self.moveMade = True
                    self.signal = False
                    self.gameOver = False

            self.manager.process_events(event)

    def __reset(self):
        self.__init__()
        print("Reset game")
