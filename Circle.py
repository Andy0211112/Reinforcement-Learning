from Parallel_self_play import parallel_self_play
from Parallel_play_with_stockfish import parallel_self_play_with_stockfish
from Train import Trainer
from Agent import ChessAgent
import os
from Config import Config
from glob import glob
import torch

if __name__ == '__main__':
    C = Config()
    C.model.res_layer_num=19
    init_model_name = ''
    pattern = os.path.join(C.model_dir, 'train_%s' % "*")
    init_model_name = list(sorted(glob(pattern)))[-1].split('/')[-1]
    print(f'best model:{init_model_name}')
    while True:
        game_datas_names = parallel_self_play(modelname=init_model_name,num_games=1000)
        print(f'Start training from: {game_datas_names}')
        T = Trainer(config=C)
        T.num_data=len(game_datas_names)
        agent = ChessAgent("w",path=init_model_name,config=C)
        init_model_name = T.train_from_data(agent,game_datas_names)
        torch.cuda.empty_cache()
    
# 
# CUDA_VISIBLE_DEVICES=1 nohup python3 Circle.py > /home/nthuuser/Andy/RL/rl_chess_softP/log/loop_log.out 2> /home/nthuuser/Andy/RL/rl_chess_softP/log/loop_log.err
# CUDA_VISIBLE_DEVICES=1 nohup python3 Circle.py > /home/nthuuser/Andy/RL/rl_chess_softP/log/self_play.out 2> /home/nthuuser/Andy/RL/rl_chess_softP/log/self_play.err
# CUDA_VISIBLE_DEVICES=1 python3 Circle.py