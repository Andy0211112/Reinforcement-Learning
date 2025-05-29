# Chess Reinforcement Learning

## Data
- `models/model_best_*.pt`: Best model files.
- `game_data/*.json`: Generated training datasets.

## How to Use

### Install Libraries
```bash
pip install -r requirements.txt
```
### Self-Play
To generate a self-play dataset using the latest model generation, use the Self_play.py or Parellel.py.

### Train
#### Reinforcement Learning
1. **Generate Dataset**: Use *Self_Play.py* to create a self-play dataset.
2. **Train Model**: Use *Train.ipynb* to train your model with the dataset generated in Step 1.
- Repeat this process iteratively to improve your model.
#### Supervised Learning
1. **Download Dataset**: Obtain chess game datasets from [FICS Games](https://www.ficsgames.org/download.html) or other sources. Convert the data into .json format (e.g., [(chess.fen, chess.Move, value), ...]).
2. **Train Model**: Place the converted dataset in the game_data folder and train your model using *Train.ipynb*.
### Evaluate
Use *Evaluate.ipynb* to:
- Play against your model.
- Observe self-play matches between versions of your model.

## References
- [Reinforcement Learning Notes](https://deepanshut041.github.io/Reinforcement-Learning/notes/00_Introduction_to_rl/)
- [Chess Deep RL GitHub Repository](https://github.com/zjeffer/chess-deep-rl)
- [Chess AlphaZero Repository](https://github.com/Zeta36/chess-alpha-zero/tree/master)
- [AlphaZero Nature Paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
- [AlphaZero Paper on arXiv](https://arxiv.org/abs/1712.01815)
- [MCTS on Wikipedia](https://zh.wikipedia.org/zh-tw/%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2)
- [FICS Games Dataset](https://www.ficsgames.org/download.html)
