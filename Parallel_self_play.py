import torch.multiprocessing as mp
from datetime import datetime
from multiprocessing import Semaphore, Manager
from Config import Config
from Agent import ChessAgent
import chess
import os
import json


class SelfPlayer:
    def __init__(self, white_model_path=None, black_model_path=None, config=None):
        self.config = config or Config()
        self.white_agent = ChessAgent("white", path=white_model_path, config=self.config)
        self.black_agent = ChessAgent("black", black_model_path, self.config)
        self.memory = []

    def save_to_file(self, memory):
        """將資料集儲存到 JSON 文件"""
        num = len(memory)
        if memory:
            D = datetime.today()
            data_name = f"self_play_dataset_{D.month:02}{D.day:02}{D.hour:02}{D.minute:02}{D.second:02}.json"
            file_path = os.path.join(
                self.config.play_data_dir,
                data_name
            )
            with open(file_path, "w") as f:
                json.dump(memory, f, indent=4)
            print(f"Saved {num} records to {file_path}")
        return data_name

    def play_games(self, games=1000, max_data_len=100000, result_queue=None, cpu_id=None):
        """Play games in a single process and save results."""
        # 綁定到特定 CPU 核心
        if cpu_id is not None:
            if hasattr(os, "sched_setaffinity"):  # Linux 設定 CPU affinity
                os.sched_setaffinity(0, {cpu_id})

        local_memory = []
        board = chess.Board()

        for _ in range(games):
            board.reset()
            game_memory = []
            current_agent = self.white_agent

            while not board.is_game_over():
                state = board.copy()
                move,policy = current_agent.choose_action(board, n_simulations=200, max_depth=10)
                board.push(chess.Move.from_uci(move))

                game_memory.append((state.fen(), policy, current_agent))
                current_agent = self.black_agent if current_agent == self.white_agent else self.white_agent

            # Determine game outcome
            if board.is_checkmate():
                value = 1 if board.turn == chess.BLACK else -1
            else:
                value = 0

            # Update memories with correct values
            for state, policy, agent in game_memory:
                local_memory.append((state, policy, value))
                value = -value  # Flip value for opponent
            
            if len(local_memory) >= max_data_len:
                self.save_to_file(local_memory)
                local_memory.clear()

        name = self.save_to_file(local_memory)  # 最後存檔

        result_queue.put(name)  # 把數據傳回主進程
        return 

def parallel_self_play(modelname='',num_games=1000):
    """使用多進程來平行運行 SelfPlayer，每個進程綁定到一個 CPU 核心"""
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    num_processes = 20
    games_per_process = num_games // num_processes
    processes = []
    D = datetime.today()
    print(f'Start self play {num_games} games at {D}, using {num_processes} cpu cores, ')
    for cpu_id in range(num_processes):
        p = mp.Process(
            target=SelfPlayer(white_model_path=modelname, black_model_path=modelname).play_games,
            args=(games_per_process, 100000, result_queue, cpu_id),
            daemon=False  # daemon 設 False，確保 queue 可以傳遞
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished, collecting file names...")  # 確保所有子進程已結束

    file_names = []
    while not result_queue.empty():
        file_names.append(result_queue.get())

    print("Collected file names:", file_names)  # 確保這時候 `file_names` 已經有值

    return file_names



if __name__ == '__main__':
    import torch 
    modelname = 'train_040103387800.pt'
    names = parallel_self_play(modelname=modelname)
    print(names)
    torch.cuda.empty_cache() # start 3/30-01:38
# CUDA_VISIBLE_DEVICES=1 nohup python3 pa.py > /home/nthuuser/Andy/RL/rl_chess_softP/log/self_play_log.out 2> /home/nthuuser/Andy/RL/rl_chess_softP/log/self_play_log.err
    