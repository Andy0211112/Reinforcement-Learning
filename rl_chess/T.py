from Config import Config
import torch
from ChessNet import ChessNet
from torchsummary import summary
from Agent import ChessAgent

from Self_play import SelfPlayer
from Train import Trainer
config = Config()

# -------------Model test------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ChessNet().to(device)
# print(summary(model, (18, 8, 8)))

# -------------selfplay----------------
# S = SelfPlayer(config=config)
# S.play_games(1)

# --------------train-----------
agent = ChessAgent("w")
T = Trainer()
play_data_filename_tmpl = "best_%s.json"
T.train_from_data(agent=agent, play_data_filename_tmpl=play_data_filename_tmpl)
