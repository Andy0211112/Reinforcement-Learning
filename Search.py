import numpy as np
import math
from Board import GameState
import chess
from Config import Config

config = Config()


class Node:
    def __init__(self, state, config=config):
        self.config = config
        self.state = state             # 棋局狀態 (GameState)
        self.children = {}             # 子節點: {action: Node}
        self.visits = 0                # 節點被訪問次數 N(s)
        self.total_value = 0           # 總回報值 W(s)
        self.policy = None             # 策略分佈 P(s, a)

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, policy, prune_threshold=0.01):
        """
        擴展節點，基於策略 P(s, a) 和全域 index_to_action 進行剪枝。
        Args:
            policy (list): 動作機率陣列，index 對應到全域變數 index_to_action。
            prune_threshold (float): 剪枝閾值，低於該值的動作將被忽略。
        """
        # self.policy = {
        #     chess.Move.from_uci(self.config.labels[i]): p for i, p in enumerate(policy) if p >= prune_threshold
        # }
        self.policy = {chess.Move.from_uci(
            self.config.labels[i]): p for i, p in enumerate(policy)
        }
        for action in self.state.get_legal_actions():
            if action in self.policy.keys():
                new_state = self.state.copy_and_apply(action)
                self.children[action] = Node(new_state)

    def select_child(self, c_puct=1.0):
        """
        使用 PUCT 演算法選擇子節點。
        Args:
            c_puct (float): 探索與利用的權衡參數。
        Returns:
            (action, Node): 最佳動作及其對應子節點。
        """
        total_visits = sum(child.visits for child in self.children.values())
        best_score, best_action, best_node = -float('inf'), None, None

        for action, child in self.children.items():
            q = (-1) * child.total_value / (child.visits + 1e-6)  # 避免除以 0 # 下個點的total value是反的
            u = c_puct * self.policy[action] * \
                math.sqrt(total_visits) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score, best_action, best_node = score, action, child

        return best_action, best_node

    def update(self, value):
        """
        更新節點的訪問次數和回報值。
        Args:
            value (float): 當前模擬的回傳值。
        """
        self.visits += 1
        self.total_value += value

    def best_action(self, temperature=1):
        """
        根據訪問次數 N(s, a) 選擇最佳行棋，支持隨機性。
        Args:
            temperature (float): 控制隨機性的參數，越接近 0 越貪婪。
        Returns:
            action: 根據隨機性或最大訪問次數選出的行棋。
        """
        visits = np.array([child.visits for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature == 0:  # 純貪婪策略
            return actions[np.argmax(visits)]

        # 應用溫度進行概率調整
        probabilities = visits ** (1 / temperature)
        probabilities /= probabilities.sum()
        if len(actions) == 0:
            return None
        
        policy = self.get_policy(actions,probabilities)

        # 隨機抽樣
        return np.random.choice(actions, p=probabilities),policy
    
    def get_policy(self,actions,probabilities):
        policy = np.zeros((self.config.n_labels))
        for action,p in zip(actions,probabilities):
            ind = self.config.all_moves2index_dict[action.uci()]
            policy[ind] = p
        return list(policy)


class MCTS:
    def __init__(self, model, max_depth=15, config=None):
        """
        初始化 MCTS。
        Args:
            get_model_p_v (function): 獲取策略 (policy) 和價值 (value) 的神經網路函數。
            max_depth (int): 每次模擬的最大深度。
        """
        self.model = model
        self.max_depth = max_depth
        self.config = config if config else Config()

    def search(self, root_state, n_simulations):
        """
        執行 MCTS 搜索。
        Args:
            root_state (chess.Board): 初始遊戲狀態。
            n_simulations (int): 模擬次數。
        Returns:
            Node: MCTS 搜索後的根節點。
        """
        root = Node(GameState(root_state.fen()))

        for _ in range(n_simulations):
            node = root
            path = []
            depth = 0

            # Selection
            while not node.is_leaf() and depth < self.max_depth:
                action, node = node.select_child()
                path.append(node)
                depth += 1

            # Expansion
            if not node.state.is_game_over() and depth < self.max_depth:
                policy, value = self.model(node.state.tensor())
                policy = policy.squeeze(0).detach().cpu().numpy()
                value = value.squeeze(0).detach().cpu().numpy()
                if node.state.is_black_turn():
                    policy = self.config.flip_policy(policy)
                node.expand(policy)
            else:
                # 如果遊戲結束或達到模擬深度，直接返回當前值
                if node.state.is_game_over():
                    value = node.state.get_result()
                    # flip value
                    if node.state.is_black_turn():
                        value = -value
                else:
                    policy, value = self.model(node.state.tensor())
                    # policy = policy.squeeze(0).detach().cpu().numpy()
                    value = value.squeeze(0).detach().cpu().numpy()
                    # if is_black_turn(node.state.fen()):
                    #     policy = flip_policy(policy)

            # Backpropagation
            for ancestor in reversed(path):
                ancestor.update(value)
                value = -value  # 翻轉值，因對手的收益是自己的損失

        return root
