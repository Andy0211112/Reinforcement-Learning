from Config import Config
import torch
from ChessNet import ChessNet
from Agent import ChessAgent

from Self_play import SelfPlayer
from Train import Trainer
import os
from glob import glob
# from ..rl_chess.Train import Trainer as old_Trainer
# -------------Model test------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ChessNet().to(device)
# print(summary(model, (18, 8, 8)))

# -------------selfplay----------------
# config = Config()
# S = SelfPlayer(config=config)
# S.play_games(1)

# --------------train-----------
# C = Config()
# init_model_name = ''
# pattern = os.path.join(C.model_dir, 'train_%s' % "*")
# init_model_name = list(sorted(glob(pattern)))[-1].split('/')[-1]
# print(f'best model:{init_model_name}')
# agent = ChessAgent(name='w',path=init_model_name)
# T = Trainer(C)
# play_data_filename_tmpl = "self_play_dataset_041812%s.json"
# # T.train_from_data(agent=agent, play_data_filename_tmpl=play_data_filename_tmpl)
# ---------------Train from old data----------------
C = Config()
init_model_name = ''
pattern = os.path.join(C.model_dir, 'train_%s' % "*")
init_model_name = list(sorted(glob(pattern)))[-1].split('/')[-1]
print(f'best model:{init_model_name}')
agent = ChessAgent(name='w',path=init_model_name)
T = Trainer(C)
T.num_data = 20
play_data_filename_tmpl = "self_play_dataset_042920%s.json"
T.train_from_data(agent=agent, play_data_filename_tmpl=play_data_filename_tmpl)

## CUDA_VISIBLE_DEVICES=1 nohup python3 T.py > /home/nthuuser/Andy/RL/rl_chess_softP/log/onlytrain_log.out 2> /home/nthuuser/Andy/RL/rl_chess_softP/log/onlytrain_log.err
