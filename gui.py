import tkinter as tk
from tkinter import messagebox
import chess
from PIL import Image, ImageTk
import torch

from model import AlphaZeroNet
from mcts import MCTS

BOARD_SIZE = 8
SQUARE_SIZE = 90
PIECE_IMAGES = {}

def load_piece_images():
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
        self.root.title("Chess AI with AlphaZero")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet().to(self.device)
        self.model.load_state_dict(torch.load("model.pt", map_location=self.device))
        self.model.eval()

        self.player_color = chess.WHITE
        self.ai_think_time = 2

        # Menu UI
        self.menu_frame = tk.Frame(self.root)
        self.menu_frame.pack()
        tk.Label(self.menu_frame, text="Chọn màu:").pack()
        self.color_var = tk.StringVar(value="white")
        tk.Radiobutton(self.menu_frame, text="Trắng", variable=self.color_var, value="white").pack()
        tk.Radiobutton(self.menu_frame, text="Đen", variable=self.color_var, value="black").pack()

        tk.Label(self.menu_frame, text="Thời gian suy nghĩ AI (giây):").pack()
        self.time_entry = tk.Entry(self.menu_frame)
        self.time_entry.insert(0, "2")
        self.time_entry.pack()

        tk.Button(self.menu_frame, text="Bắt đầu ván mới", command=self.start_game).pack(pady=10)

        # Canvas game
        self.canvas = tk.Canvas(self.root, width=BOARD_SIZE*SQUARE_SIZE, height=BOARD_SIZE*SQUARE_SIZE)
        self.canvas.bind("<Button-1>", self.on_click)

        self.board = None
        self.selected_square = None
        self.legal_destinations = []

    def start_game(self):
        color = self.color_var.get()
        self.player_color = chess.WHITE if color == "white" else chess.BLACK

        try:
            self.ai_think_time = int(self.time_entry.get())
        except ValueError:
            self.ai_think_time = 2

        self.board = chess.Board()
        self.selected_square = None
        self.legal_destinations = []

        self.menu_frame.pack_forget()
        self.canvas.pack()
        self.draw_board()

        # Nếu người chơi chọn đen, để AI đi trước
        if self.board.turn != self.player_color:
            self.root.after(500, self.ai_move)

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

        # Highlight nước đi
        for dest in self.legal_destinations:
            col = chess.square_file(dest)
            row = 7 - chess.square_rank(dest)
            x1 = col * SQUARE_SIZE
            y1 = row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=4)

    def on_click(self, event):
        if self.board.is_game_over() or self.board.turn != self.player_color:
            return

        col = event.x // SQUARE_SIZE
        row = 7 - (event.y // SQUARE_SIZE)
        square = chess.square(col, row)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
                self.legal_destinations = [
                    move.to_square for move in self.board.legal_moves if move.from_square == square
                ]
                self.draw_board()
        else:
            move = None
            for legal_move in self.board.legal_moves:
                if (legal_move.from_square == self.selected_square and
                    legal_move.to_square == square):
                    move = legal_move
                    break

            if move:
                self.board.push(move)
                self.selected_square = None
                self.legal_destinations = []
                self.draw_board()
                if self.board.is_game_over():
                    self.end_game()
                else:
                    self.root.after(500, self.ai_move)
            else:
                self.selected_square = None
                self.legal_destinations = []
                self.draw_board()

    def ai_move(self):
        if self.board.is_game_over():
            return
        mcts = MCTS(self.model, time_limit=self.ai_think_time)
        move = mcts.search(self.board)
        self.board.push(move)
        self.draw_board()
        if self.board.is_game_over():
            self.end_game()

    def end_game(self):
        result = self.board.result()
        if result == "1-0":
            message = "✅ Trắng thắng!"
        elif result == "0-1":
            message = "✅ Đen thắng!"
        else:
            message = "🤝 Hoà!"

        messagebox.showinfo("Kết thúc ván", f"Kết quả: {message}")
        self.canvas.pack_forget()
        self.menu_frame.pack()

if __name__ == "__main__":
    root = tk.Tk()
    load_piece_images()
    app = ChessGUI(root)
    root.mainloop()
