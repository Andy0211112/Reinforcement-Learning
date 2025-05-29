from Config import Config
from Agent import ChessAgent
import chess
from datetime import datetime
import os
import json


class SelfPlayer:
    def __init__(self, white_model_path=None, black_model_path=None, config=Config()):
        self.config = config
        self.device = self.config.device

        self.white_agent = ChessAgent(
            "white", path=white_model_path, config=self.config)
        self.black_agent = ChessAgent(
            "black", black_model_path, self.config)
        self.board = chess.Board()
        self.memory = []

    def save_to_file(self):
        """將資料集儲存到 JSON 文件"""
        num = len(self.memory)
        print(self.memory)
        D = datetime.today()
        if self.memory:
            file_path = os.path.join(
                self.config.play_data_dir, f"self_play_dataset_{D.month:02}{D.day:02}{D.hour:02}{D.minute:02}{D.second:02}.json")
            with open(file_path, "w") as f:
                json.dump(self.memory, f, indent=4)
            print(f"Saved {num} records to {file_path}")
            self.memory.clear()

    def play_games(self, games=1000, max_data_len=100000):
        """Play a game between the two agents and collect training data."""
        print(f"Start self play {games}.")
        for _ in range(games):
            self.board.reset()
            game_memory = []
            current_agent = self.white_agent

            while not self.board.is_game_over():
                state = self.board.copy()

                move,policy = current_agent.choose_action(
                    self.board, n_simulations=200, max_depth=10)

                # print(move)

                self.board.push(chess.Move.from_uci(move))

                # The value target will be updated after the game ends
                game_memory.append((state.fen(), policy, current_agent))
                current_agent = self.black_agent if current_agent == self.white_agent else self.white_agent

            # Determine game outcome
            if self.board.is_checkmate():
                value = 1 if self.board.turn == chess.BLACK else -1
            else:  # Draw
                value = 0

            # Update memories with correct values
            for state, policy, agent in game_memory:
                self.memory.append((state, policy, value))
                value = -value  # Flip value for opponent
            if len(self.memory) >= max_data_len:
                self.save_to_file()
        self.save_to_file()

        return value

    def test(self):
        self.board.reset()
        import time
        current_agent = self.white_agent
        while not self.board.is_game_over():
            move = current_agent.choose_action(self.board)

            self.board.push(chess.Move.from_uci(move))
            print(self.board)
            time.sleep(1)
            current_agent = self.black_agent if current_agent == self.white_agent else self.white_agent
