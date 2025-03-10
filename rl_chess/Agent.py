import os
from ChessNet import ChessNet
import torch
import torch.optim as optim
from Config import Config
from Search import MCTS
from random import random


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy, v_const=5):
        # Value loss: Mean squared error
        value_error = (value - y_value) ** 2

        # Policy loss: Cross-entropy
        y_policy = torch.clamp(y_policy, min=1e-6)  # Prevent log(0)
        policy_error = torch.sum(-policy * torch.log(y_policy), dim=1)

        # Combine losses
        total_error = value_error.squeeze()*v_const + policy_error
        return total_error.mean()


class ChessAgent:
    def __init__(self, name, path=None, config=Config()):
        self.name = name
        self.config = config
        self.device = self.config.device

        self.model = ChessNet().to(self.config.device)

        self.optimizer = optim.SGD(self.model.parameters(
        ), lr=0.0001, momentum=0.9, weight_decay=1e-4)
        self.criterion = AlphaLoss()

        self.batch_size = 512

        self.tau_decay_rate = 0.99
        if path:
            if not self.load_model(self.config.model_dir, path):
                print('path doesnt exist')
            self.model.to(self.config.device)
        else:
            pass

    def load_model(self, dir, filename):
        """Load a model from disk."""
        if os.path.exists(os.path.join(dir, filename)):
            model_state = torch.load(os.path.join(
                dir, filename), map_location=self.device)
            self.model.load_state_dict(model_state)
            print(f"Model loaded from {filename}")
            return True
        return False

    def save_model(self, dir, filename):
        """Save a model to disk."""
        path = os.path.join(dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def choose_action(self, board, max_depth=15, n_simulations=200, temperature=1.0, mode='mcts'):
        """Choose action by probability."""
        if mode == 'mcts':
            tree = MCTS(self.model, max_depth=max_depth, config=self.config)
            root = tree.search(
                board, n_simulations=n_simulations)
            move = root.best_action(temperature=temperature)
            if move is None:
                move = random.choice(list(board.legal_moves))
            return move.uci()
