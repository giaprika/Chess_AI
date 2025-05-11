import chess
import pygame
from game_init import GameInit
from config import *
from utils import logGameStatus

class PlayMode(GameInit):

    def __init__(self):
        super().__init__()

    def mainLoop(self):
        while self.running:
            self.time_delta = self.clock.tick(MAX_FPS) / 1000
            self.__eventHandler()

            if self.moveMade:
                if self.gs.board.is_game_over():
                    self.gameOver = True
                self.editChessPanel()
                self.moveMade = False

            self.manager.update(self.time_delta)
            self.drawGameScreen()
            pygame.display.update()

            if self.gameOver:
                self.drawGameScreen()

    def __eventHandler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Game Quit")
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.clickUserHandler()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.__reset()
                elif event.key == pygame.K_z:
                    self.gs.undoMove()
                    self.moveMade = True
                    self.gameOver = False
                elif event.key == pygame.K_u:
                    logGameStatus(self.gs.piece_ingame)

            self.manager.process_events(event)

    def __reset(self):
        self.__init__()
        print("Reset game")
