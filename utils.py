import os
import numpy as np
import chess
import torch

def get_filename_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def get_loss_from_model_name(file_name):
    try:
        loss_str = file_name.split('_')[-1]
        return float(loss_str)
    except ValueError:
        raise ValueError(f"Filename '{file_name}' does not contain valid loss value.")

def board_to_tensor(board):
    tensor = np.zeros((20, 8, 8), dtype=np.float32)
    
    # Piece channels
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            offset = 0 if piece.color == chess.WHITE else 6
            channel = piece_map[piece.piece_type] + offset
            row, col = divmod(square, 8)
            tensor[channel, row, col] = 1

    # Repetition history
    tensor[12, :, :] = float(board.is_repetition(2))
    tensor[13, :, :] = float(board.is_repetition(3))

    
    # Castling rights
    tensor[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    tensor[17, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    
    # En passant
    if board.ep_square:
        row, col = divmod(board.ep_square, 8)
        tensor[18, row, col] = 1.0

    
    # Turn
    tensor[19, :, :] = 1 if board.turn == chess.WHITE else 0
    
    return torch.tensor(tensor)

def move_to_index(move):
    if not isinstance(move, chess.Move):
        raise TypeError(f"Expected chess.Move, got {type(move)}")

    from_sq = move.from_square
    to_sq = move.to_square
    dx = chess.square_file(to_sq) - chess.square_file(from_sq)
    dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    # Queen-like moves (8 directions x 7 distances)
    if abs(dx) == abs(dy) or dx == 0 or dy == 0:
        direction = (np.sign(dx), np.sign(dy))
        distance = max(abs(dx), abs(dy))
        directions = [(1,0), (1,1), (0,1), (-1,1),
                      (-1,0), (-1,-1), (0,-1), (1,-1)]
        if direction in directions and 1 <= distance <= 7:
            dir_idx = directions.index(direction)
            move_type = dir_idx * 7 + (distance - 1)  # 0–55
            return from_sq * 73 + move_type

    # Knight moves
    knight_deltas = [(2,1), (1,2), (-1,2), (-2,1),
                     (-2,-1), (-1,-2), (1,-2), (2,-1)]
    if (dx, dy) in knight_deltas:
        move_type = 56 + knight_deltas.index((dx, dy))  # 56–63
        return from_sq * 73 + move_type

    # Underpromotion
    if move.promotion and move.promotion != chess.QUEEN:
        # 3 types × 3 directions = 9
        promo_map = {
            (0, 1): 0,   # forward
            (-1, 1): 1,  # left capture
            (1, 1): 2    # right capture
        }
        promo_piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK].index(move.promotion)
        dir_idx = promo_map.get((dx, dy))
        if dir_idx is not None:
            move_type = 64 + promo_piece * 3 + dir_idx  # 64–72
            return from_sq * 73 + move_type

    # Queen promotion handled same as normal move (included above)
    raise ValueError(f"Unrecognized move pattern: {move}")

def index_to_move(board, index):
    for move in board.legal_moves:
        if move_to_index(move) == index:
            return move
    return None  # hoặc raise Exception nếu không tìm thấy


def get_policy_vector(board, mcts_node):
    policy = np.zeros(4672, dtype=np.float32)
    total_visits = sum(child.visit_count for child in mcts_node.children.values())
    
    for move, child in mcts_node.children.items():
        idx = move_to_index(move)
        policy[idx] = child.visit_count / total_visits
    
    return policy