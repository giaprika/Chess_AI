import chess.pgn
import chess.engine
import numpy as np
import os
import torch
from utils import board_to_tensor, move_to_index  # Đảm bảo các hàm này đã được định nghĩa
from multiprocessing import Process
from replay_buffer import save_buffer

TIME_LIMIT = 3.0
games_per_chunk = 5

def get_stockfish_policy(board, engine, time_limit=TIME_LIMIT, top_k=20):
    policy = np.zeros(4672, dtype=np.float32)
    try:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return policy

        multipv = min(top_k, len(legal_moves))
        info = engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=multipv)

        scores = []
        moves = []
        for entry in info:
            move = entry.get("pv", [None])[0]
            if move not in legal_moves:
                continue
            wdl = entry["score"].wdl()
            pov_wdl = wdl.pov(board.turn)
            total = pov_wdl.wins + pov_wdl.draws + pov_wdl.losses
            win_prob = pov_wdl.wins / total if total > 0 else 0.0
            scores.append(win_prob)
            moves.append(move)

        if not scores:
            return policy

        scores = np.array(scores, dtype=np.float32)
        temperature = 0.5
        probs = np.exp(scores / temperature)
        probs = probs / probs.sum()

        for move, prob in zip(moves, probs):
            try:
                idx = move_to_index(move)
                if 0 <= idx < 4672:
                    policy[idx] = prob
            except Exception as e:
                print(f"[ERROR] move_to_index({move}): {e}")
    except Exception as e:
        print(f"[Lỗi phân tích Stockfish]: {e}")
    
    return policy

def get_stockfish_value(board, engine, time_limit=TIME_LIMIT):
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        wdl = info["score"].wdl()
        pov_wdl = wdl.pov(board.turn)
        value = pov_wdl.wins - pov_wdl.losses
        value /= 1000.0
        return float(value)
    except Exception as e:
        print(f"[Lỗi lấy value]: {e}")
        try:
            score = info["score"].relative.score(mate_score=10000)
            return np.tanh(score / 400.0)
        except:
            return 0.0

def process_pgn_with_stockfish(pgn_path, save_dir, stockfish_path):
    os.makedirs(save_dir, exist_ok=True)
    progress_path = os.path.join(save_dir, "progress.txt")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    chunk_data = []
    game_count = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as pf:
            try:
                game_count = int(pf.read().strip())
            except:
                game_count = 0
    skip_count = game_count  # số lượng game cần bỏ qua ban đầu
    current_game_index = 0
    chunk_idx = game_count / 5

    with open(pgn_path, 'r', encoding='utf-8') as f:
        while True:
            game = chess.pgn.read_game(f)
            if not game:
                break
            if current_game_index < skip_count:
                current_game_index += 1
                continue
            print(f'Start from game {current_game_index}')
            try:
                board = game.board()
                game_data = []
                print(f'Đang xử lý game {game_count + 1}')

                for move in game.mainline_moves():
                    policy = get_stockfish_policy(board, engine)
                    value = get_stockfish_value(board, engine)
                    state = board_to_tensor(board)
                    print(f'policy: {policy}')
                    print(f'sum policy: {sum(policy)}')
                    print(f'num 1 in policy {np.sum(policy == 1)}')
                    print(f'num >0.9 in policy {np.sum(policy >= 0.9)}')
                    print(f'num >0.5 in policy: {np.sum(policy >= 0.5)}')
                    print(f'num <0.05 in policy: {np.sum(policy <= 0.05)}')
                    print(f'value: {value}')
                    game_data.append((state, policy, value))
                    board.push(move)

                chunk_data.extend(game_data)
                game_count += 1
                print(f"[{os.path.basename(pgn_path)}] Đã xử lý ván {game_count}")

                if game_count % games_per_chunk == 0:
                    save_chunk(chunk_data, save_dir, chunk_idx)
                    with open(progress_path, 'w') as pf:
                        pf.write(str(game_count))
                    chunk_idx += 1
                    chunk_data = []

            except Exception as e:
                print(f"Lỗi ở ván {game_count}: {e}")

    if chunk_data:
        save_chunk(chunk_data, save_dir, chunk_idx)

    engine.quit()
    print(f"[{os.path.basename(pgn_path)}] Hoàn thành!")

def save_chunk(chunk_data, save_dir, chunk_idx):
    states, policies, values = zip(*chunk_data)
    file_path = os.path.join(save_dir, f"chunk_{chunk_idx:04d}.pt")
    data = [(s, p, v) for s, p, v in zip(states, policies, values)]
    save_buffer(data, file_path)
    print(f"Đã lưu chunk {chunk_idx} vào {file_path}")

def run_in_parallel(pgn_paths, save_dir, stockfish_path):
    processes = []
    for i, pgn_path in enumerate(pgn_paths):
        file_save_dir = os.path.join(save_dir, f"file_{i+7}")
        p = Process(target=process_pgn_with_stockfish, args=(pgn_path, file_save_dir, stockfish_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    print("Start đa tiến trình...")
    pgn_files = [
        "./lichess_db_standard_rated_2013-07.pgn",
        "./lichess_db_standard_rated_2013-08.pgn",
        "./lichess_db_standard_rated_2013-09.pgn",
        "./lichess_db_standard_rated_2013-10.pgn",
        "./lichess_db_standard_rated_2013-11.pgn",
        "./lichess_db_standard_rated_2013-12.pgn"
    ]

    run_in_parallel(
        pgn_paths=pgn_files,
        save_dir="train_data",
        stockfish_path="./stockfish/stockfish-windows-x86-64-avx2.exe"
    )