import pygame
import pygame_gui
import torch
import chess

from config import *
from model import AlphaZeroNet
from utils import board_to_tensor
from chess_engine import GameState

class GameInit:
    def __init__(self):
        self.screen = pygame.display.get_surface()
        self.IMAGES = {}

        self.__loadImages()
        self.image_black_win = pygame.transform.scale(pygame.image.load('./data/images/black_win.jpg'),
                                                      (2.62 * SQ_SIZE, SQ_SIZE))
        self.image_black_win.set_alpha(200)
        self.image_white_win = pygame.transform.scale(pygame.image.load('./data/images/white_win.jpg'),
                                                      (2.62 * SQ_SIZE, SQ_SIZE))
        self.image_white_win.set_alpha(200)

        self.__loadSound()

        # Surface for board game
        self.background = pygame.Surface((WIDTH_WINDOW_AI, HEIGHT_WINDOW_AI))
        self.background.fill((32, 32, 32))

        # GUI
        self.manager = pygame_gui.UIManager((WIDTH_WINDOW, HEIGHT_WINDOW), theme_path="./data/theme_custom.json")
        self.__loadGUI()

        self.gs = GameState()
        self.clock = pygame.time.Clock()

        self.click = ()
        self.playerClicks = []
        self.moveMade = False
        self.validMoves = self.gs.getValidMoves()
        self.running = True
        self.gameOver = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model("data/model.pt")

        self.editChessPanel()

    def load_model(self, model_path):
        self.model = AlphaZeroNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def __loadGUI(self):

        self.chess_panel = pygame_gui.elements.UIWindow(
            rect=pygame.Rect((LEFT_PANEL, TOP_PANEL), (WIDTH_PANEL, HEIGHT_PANEL)),
            manager=self.manager,
            window_display_title='Panel chess'
        )

        self.text_box = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((LEFT_MOVE_BOX, TOP_MOVE_BOX), (WIDTH_MOVE_BOX, HEIGHT_MOVE_BOX)),
            html_text='',
            manager=self.manager,
            container=self.chess_panel
        )

        self.label_turn = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((LEFT_TURN, TOP_TURN), (WIDTH_LABEL, HEIGHT_LABEL)),
            text='Turn: ', manager=self.manager,
            container=self.chess_panel
        )

        self.label_possible_move = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((LEFT_POSSIBLE_MOVE, TOP_POSSIBLE_MOVE), (WIDTH_LABEL, HEIGHT_LABEL)),
            text='Possible moves: ', manager=self.manager,
            container=self.chess_panel
        )

        self.label_incheck = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((LEFT_INCHECK, TOP_INCHECK), (WIDTH_LABEL, HEIGHT_LABEL)),
            text='In Check : ', manager=self.manager,
            container=self.chess_panel
        )

    def __loadImages(self):
        pieces = ['wP', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bP', 'bR', 'bN', 'bB', 'bK', 'bQ']
        for piece in pieces:
            image_path = f"data/images/chess/{piece}.png"  # Đảm bảo đúng đường dẫn và tên hình ảnh
            try:
                self.IMAGES[piece] = pygame.transform.scale(
                    pygame.image.load(image_path), (SQ_SIZE, SQ_SIZE))  # Tải và điều chỉnh kích thước hình ảnh
            except pygame.error:
                print(f"Error loading image: {image_path}")  # In ra lỗi nếu không tải được hình ảnh

    def __loadSound(self):
        self.sound_move = pygame.mixer.Sound('./data/sound/move.wav')
        self.sound_capture = pygame.mixer.Sound('./data/sound/capture.wav')

    def drawGameScreen(self):
        self.screen.blit(self.background, (0, 0))
        self.drawBoard()
        self.drawLastMove()
        self.drawPiece()
        self.highlightSquares()
        self.screen.blit(self.screen, (0, 0))
        self.manager.draw_ui(self.screen)
        if self.gameOver:
            self.drawGameOver()

    def drawLastMove(self):
        if len(self.gs.board.move_stack) > 0:
            lastMove = self.gs.board.peek()
            from_sq = lastMove.from_square
            to_sq = lastMove.to_square
            surface = pygame.Surface((SQ_SIZE, SQ_SIZE))
            surface.set_alpha(100)
            surface.fill((153, 255, 255))
            for square in [from_sq, to_sq]:
                rank = 7 - chess.square_rank(square)
                file = chess.square_file(square)
                self.screen.blit(surface, (file * SQ_SIZE, rank * SQ_SIZE))

    def drawBoard(self):
        for i in range(DIMENSION):
            for j in range(DIMENSION):
                color = colorBoard[(i + j) % 2]
                x = i * SQ_SIZE
                y = j * SQ_SIZE
                pygame.draw.rect(self.screen, color, pygame.Rect(x, y, SQ_SIZE, SQ_SIZE))

    def drawPiece(self):
        for i in range(DIMENSION):
            for j in range(DIMENSION):
                piece = self.gs.board.piece_at(chess.square(j, 7 - i))
                if piece:
                    # Kiểm tra và chuyển đổi ký hiệu thành định dạng trong self.IMAGES
                    if piece.color == chess.WHITE:
                        piece_key = f"w{piece.symbol()}"  # Tạo khóa cho quân trắng, ví dụ: 'wP', 'wR'
                    else:
                        piece_key = f"b{piece.symbol().upper()}"  # Tạo khóa cho quân đen, ví dụ: 'bp', 'br'

                    # Kiểm tra xem quân cờ có tồn tại trong self.IMAGES không
                    if piece_key in self.IMAGES:
                        piece_image = self.IMAGES[piece_key]
                        self.screen.blit(piece_image, pygame.Rect(j * SQ_SIZE, i * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def drawGameOver(self):
        if self.gs.board.result() == "1-0":
            self.screen.blit(self.image_white_win, (191, 223))
        elif self.gs.board.result() == "0-1":
            self.screen.blit(self.image_black_win, (191, 223))
        # else:
        #     self.screen.blit(self.image_draw, (191, 223))  # Nếu bạn có hình hòa

    def highlightSquares(self):
        sqHighlight = []
        if self.click:
            from_square = chess.square(self.click[1], 7 - self.click[0])
            for move in self.gs.board.legal_moves:
                if move.from_square == from_square:
                    sqHighlight.append(move.to_square)

            for square in sqHighlight:
                rank = 7 - chess.square_rank(square)
                file = chess.square_file(square)
                surface = pygame.Surface((SQ_SIZE, SQ_SIZE))
                surface.set_alpha(150)
                surface.fill(colorBoard[2])
                self.screen.blit(surface, (file * SQ_SIZE, rank * SQ_SIZE))

    def editChessPanel(self):
        self.text_box.set_text(self.gs.getMoveNotation())
        self.label_turn.set_text(self.gs.getTurn())
        self.label_possible_move.set_text(f'Possible moves: {len(self.validMoves)}')
        self.label_incheck.set_text(f'In Check : {self.gs.isInCheck()}')

    def clickUserHandler(self):
        pos = pygame.mouse.get_pos()
        col = int(pos[0] / SQ_SIZE)
        row = int(pos[1] / SQ_SIZE)

        if row in range(8) and col in range(8):
            square = chess.square(col, 7 - row)
            piece = self.gs.board.piece_at(square)

            # Nếu người chơi chưa chọn hoặc chọn lại chính ô đó, reset chọn
            if self.click == (row, col) or (not self.click and (piece is None or piece.color != self.gs.board.turn)):
                self.click = ()
                self.playerClicks = []
            else:
                self.click = (row, col)
                self.playerClicks.append(self.click)

            # Nếu người chơi đã chọn hai ô, thực hiện nước đi
            if len(self.playerClicks) == 2:
                from_row, from_col = self.playerClicks[0]
                to_row, to_col = self.playerClicks[1]

                from_sq = chess.square(from_col, 7 - from_row)
                to_sq = chess.square(to_col, 7 - to_row)

                move = chess.Move(from_sq, to_sq)

                # Nếu là tốt đến hàng cuối, phong hậu
                piece = self.gs.board.piece_at(from_sq)
                if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
                    move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

                if move in self.gs.board.legal_moves:
                    is_capture = self.gs.board.is_capture(move)
                    self.gs.board.push(move)
                    pygame.mixer.Sound.play(self.sound_capture if is_capture else self.sound_move)
                    self.moveMade = True

                # Reset click và cập nhật highlight
                self.click = ()
                self.playerClicks = []

                # Cập nhật lại các ô có thể di chuyển cho người chơi
                self.validMoves = list(self.gs.board.legal_moves)
                self.highlightSquares()  # Vẽ lại highlight

    def make_ai_move(self):
        if not self.model or self.gs.board.is_game_over():
            return

        x = board_to_tensor(self.gs.board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.model(x)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        best_move = self._select_best_move(self.gs.board, policy)
        if best_move:
            captured = self.gs.board.is_capture(best_move)
            self.gs.board.push(best_move)
            pygame.mixer.Sound.play(self.sound_capture if captured else self.sound_move)
            self.moveMade = True

    def _select_best_move(self, board, policy):
        legal_moves = list(board.legal_moves)
        move_scores = []

        for move in legal_moves:
            idx = self.gs.uciToPolicyIndex(move.uci())
            if idx is not None and idx < len(policy):
                move_scores.append((policy[idx], move))

        if not move_scores:
            return None

        move_scores.sort(reverse=True, key=lambda x: x[0])
        return move_scores[0][1]