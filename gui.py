import tkinter as tk
from tkinter import messagebox
import chess
from PIL import Image, ImageTk
import torch

from model import AlphaZeroNet
from mcts import MCTS

BOARD_SIZE = 8
SQUARE_SIZE = 60
PIECE_IMAGES = {}

# Load hình ảnh quân cờ
def load_piece_images():
    global PIECE_IMAGES
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K',
              'p', 'n', 'b', 'r', 'q', 'k']
    for p in pieces:
        try:
            img = Image.open(f'images/{p.lower()}{"w" if p.isupper() else "b"}.png')
            img = img.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
            PIECE_IMAGES[p] = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"⚠️ Không thể tải ảnh {p}: {e}")

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess AI GUI (AlphaZero)")

        self.canvas = tk.Canvas(root, width=BOARD_SIZE * SQUARE_SIZE,
                                      height=BOARD_SIZE * SQUARE_SIZE)
        self.canvas.pack()

        self.board = chess.Board()
        self.selected_square = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet().to(self.device)
        self.model_path = "model.pt"
        if torch.cuda.is_available():
            print("✅ Using GPU")
        else:
            print("⚠️ Using CPU")

        if torch.load:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        self.draw_board()

        self.canvas.bind("<Button-1>", self.on_click)

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#F0D9B5", "#B58863"]

        for row in range(8):
            for col in range(8):
                x1 = col * SQUARE_SIZE
                y1 = row * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

                square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                if piece:
                    image = PIECE_IMAGES.get(piece.symbol())
                    if image:
                        self.canvas.create_image(x1, y1, anchor="nw", image=image)

    def on_click(self, event):
        col = event.x // SQUARE_SIZE
        row = 7 - (event.y // SQUARE_SIZE)
        square = chess.square(col, row)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                self.root.after(500, self.ai_move)
            else:
                self.selected_square = None

    def ai_move(self):
        if self.board.is_game_over():
            messagebox.showinfo("Kết thúc", f"Kết quả: {self.board.result()}")
            return

        print("🤖 AI đang suy nghĩ...")
        mcts = MCTS(self.model, time_limit=2)
        move = mcts.search(self.board)
        print(f"🤖 AI chọn: {move}")
        self.board.push(move)
        self.draw_board()

        if self.board.is_game_over():
            messagebox.showinfo("Kết thúc", f"Kết quả: {self.board.result()}")

if __name__ == "__main__":
    root = tk.Tk()  # 🟢 Tạo root window trước khi load ảnh
    load_piece_images()
    gui = ChessGUI(root)
    root.mainloop()
