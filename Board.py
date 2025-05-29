import chess
import Environment
import numpy as np
from torch import FloatTensor
from Config import Config

config = Config()


class GameState(chess.Board):
    def __init__(self, fen=None, config=config):
        """
        初始化 GameState，默認使用標準起始局面。
        Args:
            fen (str, optional): 用於初始化的 FEN 字符串。
        """
        super().__init__(fen)
        self.config = config

    def get_legal_actions(self):
        """
        獲取當前局面下的所有合法動作。
        Returns:
            list[chess.Move]: 一個包含所有合法行棋的列表。
        """
        return list(self.legal_moves)

    def copy_and_apply(self, action):
        """
        創建當前局面的副本，並應用指定的動作。
        Args:
            action (chess.Move): 要執行的動作。
        Returns:
            GameState: 更新後的新局面。
        """
        new_state = GameState(self.fen())
        new_state.push(action)
        return new_state

    def is_game_over(self):
        """
        判斷遊戲是否結束。
        Returns:
            bool: 如果遊戲結束則返回 True，否則 False。
        """
        return super().is_game_over()

    def get_result(self):
        """
        獲取當前局面的結果。
        Returns:
            float: 
                - 1.0: 白方勝利
                - -1.0: 黑方勝利
                - 0.0: 平局
        """
        if self.is_checkmate():
            return 1.0 if self.turn == chess.BLACK else -1.0
        # elif self.is_stalemate() or self.is_insufficient_material():
        #     return 0.0
        else:
            return 0.0
        return None  # 遊戲尚未結束

    def fen(self):
        """
        返回當前局面的 FEN 表示。
        Returns:
            str: FEN 字符串。
        """
        return super().fen()

    def tensor(self):
        return FloatTensor(np.array(Environment.canon_input_planes(self.fen()))).unsqueeze(0).to(self.config.device)

    def is_black_turn(self):
        return self.fen().split(" ")[1] == 'b'
