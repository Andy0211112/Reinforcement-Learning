from glob import glob
import os
import json
from Environment import is_black_turn, canon_input_planes
import numpy as np
from Config import Config


def find_pgn_files(directory, pattern='*.pgn'):
    dir_pattern = os.path.join(directory, pattern)
    files = list(sorted(glob(dir_pattern)))
    return files


def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        print(e)


def get_game_data_filenames(play_data_dir, play_data_filename_tmpl):
    pattern = os.path.join(play_data_dir, play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


def testeval(fen, absolute=False) -> float:
    # somehow it doesn't know how to keep its queen
    piece_vals = {'K': 3, 'Q': 14, 'R': 5, 'B': 3.25, 'N': 3, 'P': 1}
    ans = 0.0
    tot = 0
    for c in fen.split(' ')[0]:
        if not c.isalpha():
            continue

        if c.isupper():
            ans += piece_vals[c]
            tot += piece_vals[c]
        else:
            ans -= piece_vals[c.upper()]
            tot += piece_vals[c.upper()]
    v = ans/tot
    if not absolute and is_black_turn(fen):
        v = -v
    assert abs(v) < 1
    return np.tanh(v * 3)  # arbitrary


config = Config()


def convert_to_cheating_data(data, config=config):
    """
    :param data: format is SelfPlayWorker.buffer
    :return:
    """
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:

        state_planes = canon_input_planes(state_fen)
        # policy = config.all_moves2index_dict[policy]
        # temp = np.zeros((1968,))
        # temp[policy] = 1
        # policy = temp
        # del temp
        if is_black_turn(state_fen):
            policy = config.flip_policy(policy)

        # move_number = int(state_fen.split(' ')[5])
        # # reduces the noise of the opening... plz train faster
        # value_certainty = min(10, move_number)/10
        # _value = value*value_certainty + testeval(state_fen, False)*(1-value_certainty)

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(value)

    return np.array(state_list, dtype=np.float32), np.array(policy_list, dtype=np.float32), np.array(value_list, dtype=np.float32)


def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    if data is None:
        return None, None, None
    return convert_to_cheating_data(data)
