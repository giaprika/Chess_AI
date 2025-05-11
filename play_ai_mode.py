import chess
import pygame
import pygame_gui.elements
import torch
from model import AlphaZeroNet
from utils import board_to_tensor, move_to_index
from game_init import GameInit
from config import *
from mcts import MCTS
from evalute_elo import model_get_best_move

class PlayAIMode(GameInit):

    def __init__(self):
        super().__init__()
        self.aiTurn = 'b'
        self.screen = pygame.display.set_mode((WIDTH_WINDOW_AI, HEIGHT_WINDOW_AI))
        self.manager.set_window_resolution((WIDTH_WINDOW_AI, HEIGHT_WINDOW_AI))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet().to(self.device)
        self.model.load_state_dict(torch.load("./data/model.pt", map_location=self.device))
        self.model.eval()

        self.mcts = MCTS(self.model, 2.0)

        self.total_nodes_log = []
        self.executionTime_log = []
        self.signal = True

        self.human_turn = True
        self.ai_thinking = False
        self.isEndGame = False

    def mainLoop(self):
        while self.running:
            self.time_delta = self.clock.tick(MAX_FPS) / 1000
            self.human_turn = self.gs.board.turn != (self.aiTurn == 'w')
            self.__eventHandler()

            if not self.human_turn and self.signal and not self.gameOver:
                self.__AIMove()

            if self.moveMade:
                self.validMoves = self.gs.getValidMoves()
                if self.gs.board.is_game_over():
                    self.gameOver = True
                self.moveMade = False
                self.signal = True
                self.editChessPanel()

            self.manager.update(self.time_delta)
            self.drawGameScreen()
            pygame.display.update()
            if self.gameOver:
                self.drawGameScreen()

    def __AIMove(self):
        # Lấy trạng thái bàn cờ từ GameState
        board = self.gs.board

        best_move = model_get_best_move(board)

        if best_move:
            captured_piece = board.piece_at(best_move.to_square)
            self.gs.makeMove(best_move)

            if captured_piece is None:
                pygame.mixer.Sound.play(self.sound_move)
            else:
                pygame.mixer.Sound.play(self.sound_capture)

            self.moveMade = True

    def __eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game Quit")
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.human_turn:
                    self.clickUserHandler()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.__reset()
                elif event.key == pygame.K_z:
                    self.gs.undoMove()
                    self.moveMade = True
                    self.signal = False
                    self.gameOver = False

            self.manager.process_events(event)

    def __reset(self):
        self.__init__()
        print("Reset game")
