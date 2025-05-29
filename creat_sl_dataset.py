"""
Contains the worker for training the model using recorded game data rather than self-play
"""
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from logging import getLogger
from threading import Thread
from time import time
from collections import defaultdict
import json
from glob import glob
import chess.pgn
from threading import Lock
from Config import Config
import enum
def start(config: Config):
    return SupervisedLearningWorker(config).start()

class ChessEnv:
    """
    Represents a chess environment where a chess game is played/

    Attributes:
        :ivar chess.Board board: current board state
        :ivar int num_halfmoves: number of half moves performed in total by each player
        :ivar Winner winner: winner of the game
        :ivar boolean resigned: whether non-winner resigned
        :ivar str result: str encoding of the result, 1-0, 0-1, or 1/2-1/2
    """

    def __init__(self):
        self.board = None
        self.num_halfmoves = 0
        self.winner = None  # type: Winner
        self.resigned = False
        self.result = None

    def reset(self):
        """
        Resets to begin a new game
        :return ChessEnv: self
        """
        self.board = chess.Board()
        self.num_halfmoves = 0
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        """
        Like reset, but resets the position to whatever was supplied for board
        :param chess.Board board: position to reset to
        :return ChessEnv: self
        """
        self.board = chess.Board(board)
        self.winner = None
        self.resigned = False
        return self

    @property
    def done(self):
        return self.winner is not None

    @property
    def white_won(self):
        return self.winner == Winner.white

    @property
    def white_to_move(self):
        return self.board.turn == chess.WHITE

    def step(self, action: str, check_over=True):
        """

        Takes an action and updates the game state

        :param str action: action to take in uci notation
        :param boolean check_over: whether to check if game is over
        """
        if check_over and action is None:
            self._resign()
            return

        self.board.push_uci(action)

        self.num_halfmoves += 1

        if check_over and self.board.result(claim_draw=True) != "*":
            self._game_over()

    def _game_over(self):
        if self.winner is None:
            self.result = self.board.result(claim_draw=True)
            if self.result == '1-0':
                self.winner = Winner.white
            elif self.result == '0-1':
                self.winner = Winner.black
            else:
                self.winner = Winner.draw

    def _resign(self):
        self.resigned = True
        if self.white_to_move:  # WHITE RESIGNED!
            self.winner = Winner.black
            self.result = "0-1"
        else:
            self.winner = Winner.white
            self.result = "1-0"

    def adjudicate(self):
        score = self.testeval(absolute=True)
        if abs(score) < 0.01:
            self.winner = Winner.draw
            self.result = "1/2-1/2"
        elif score > 0:
            self.winner = Winner.white
            self.result = "1-0"
        else:
            self.winner = Winner.black
            self.result = "0-1"

    def ending_average_game(self):
        self.winner = Winner.draw
        self.result = "1/2-1/2"

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()

    def deltamove(self, fen_next):
        moves = list(self.board.legal_moves)
        for mov in moves:
            self.board.push(mov)
            fee = self.board.fen()
            self.board.pop()
            if fee == fen_next:
                return mov.uci()
        return None

    def replace_tags(self):
        return replace_tags_board(self.board.fen())

    def canonical_input_planes(self):
        """

        :return: a representation of the board using an (18, 8, 8) shape, good as input to a policy / value network
        """
        return canon_input_planes(self.board.fen())

    def testeval(self, absolute=False) -> float:
        return testeval(self.board.fen(), absolute)


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


def check_current_planes(realfen, planes):
    cur = planes[0:12]
    assert cur.shape == (12, 8, 8)
    fakefen = ["1"] * 64
    for i in range(12):
        for rank in range(8):
            for file in range(8):
                if cur[i][rank][file] == 1:
                    assert fakefen[rank * 8 + file] == '1'
                    fakefen[rank * 8 + file] = pieces_order[i]

    castling = planes[12:16]
    fiftymove = planes[16][0][0]
    ep = planes[17]

    castlingstring = ""
    for i in range(4):
        if castling[i][0][0] == 1:
            castlingstring += castling_order[i]

    if len(castlingstring) == 0:
        castlingstring = '-'

    epstr = "-"
    for rank in range(8):
        for file in range(8):
            if ep[rank][file] == 1:
                epstr = coord_to_alg((rank, file))

    realfen = maybe_flip_fen(realfen, flip=is_black_turn(realfen))
    realparts = realfen.split(' ')
    assert realparts[1] == 'w'
    assert realparts[2] == castlingstring
    assert realparts[3] == epstr
    assert int(realparts[4]) == fiftymove
    # realparts[5] is the fifty-move clock, discard that
    return "".join(fakefen) == replace_tags_board(realfen)


def canon_input_planes(fen):
    """

    :param fen:
    :return : (18, 8, 8) representation of the game state
    """
    fen = maybe_flip_fen(fen, is_black_turn(fen))
    return all_input_planes(fen)


def all_input_planes(fen):
    current_aux_planes = aux_planes(fen)

    history_both = to_planes(fen)

    ret = np.vstack((history_both, current_aux_planes))
    assert ret.shape == (18, 8, 8)
    return ret


def maybe_flip_fen(fen, flip=False):
    if not flip:
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')

    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a

    def swapall(aa):
        return "".join([swapcase(a) for a in aa])
    return "/".join([swapall(row) for row in reversed(rows)]) \
        + " " + ('w' if foo[1] == 'b' else 'b') \
        + " " + "".join(sorted(swapall(foo[2]))) \
        + " " + foo[3] + " " + foo[4] + " " + foo[5]


def aux_planes(fen):
    foo = fen.split(' ')

    en_passant = np.zeros((8, 8), dtype=np.float32)
    if foo[3] != '-':
        eps = alg_to_coord(foo[3])
        en_passant[eps[0]][eps[1]] = 1

    fifty_move_count = int(foo[4])
    fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)

    castling = foo[2]
    auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32),
                        np.full((8, 8), int('Q' in castling),
                                dtype=np.float32),
                        np.full((8, 8), int('k' in castling),
                                dtype=np.float32),
                        np.full((8, 8), int('q' in castling),
                                dtype=np.float32),
                        fifty_move,
                        en_passant]

    ret = np.asarray(auxiliary_planes, dtype=np.float32)
    assert ret.shape == (6, 8, 8)
    return ret

# FEN board is like this:
# a8 b8 .. h8
# a7 b7 .. h7
# .. .. .. ..
# a1 b1 .. h1
#
# FEN string is like this:
#  0  1 ..  7
#  8  9 .. 15
# .. .. .. ..
# 56 57 .. 63

# my planes are like this:
# 00 01 .. 07
# 10 11 .. 17
# .. .. .. ..
# 70 71 .. 77
#


def alg_to_coord(alg):
    rank = 8 - int(alg[1])        # 0-7
    file = ord(alg[0]) - ord('a')  # 0-7
    return rank, file


def coord_to_alg(coord):
    letter = chr(ord('a') + coord[1])
    number = str(8 - coord[0])
    return letter + number


def to_planes(fen):
    board_state = replace_tags_board(fen)
    pieces_both = np.zeros(shape=(12, 8, 8), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            v = board_state[rank * 8 + file]
            if v.isalpha():
                pieces_both[ind[v]][rank][file] = 1
    assert pieces_both.shape == (12, 8, 8)
    return pieces_both


def replace_tags_board(board_san):
    board_san = board_san.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    return board_san.replace("/", "")


def is_black_turn(fen):
    return fen.split(" ")[1] == 'b'
class VisitStats:
    """
    Holds information for use by the AGZ MCTS algorithm on all moves from a given game state (this is generally used inside
    of a defaultdict where a game state in FEN format maps to a VisitStats object).
    Attributes:
        :ivar defaultdict(ActionStats) a: known stats for all actions to take from the the state represented by
            this visitstats.
        :ivar int sum_n: sum of the n value for each of the actions in self.a, representing total
            visits over all actions in self.a.
    """

    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ActionStats:
    """
    Holds the stats needed for the AGZ MCTS algorithm for a specific action taken from a specific state.

    Attributes:
        :ivar int n: number of visits to this action by the algorithm
        :ivar float w: every time a child of this action is visited by the algorithm,
            this accumulates the value (calculated from the value network) of that child. This is modified
            by a virtual loss which encourages threads to explore different nodes.
        :ivar float q: mean action value (total value from all visits to actions
            AFTER this action, divided by the total number of visits to this action)
            i.e. it's just w / n.
        :ivar float p: prior probability of taking this action, given
            by the policy network.

    """

    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0
class ChessPlayer:
    """
    Plays the actual game of chess, choosing moves based on policy and value network predictions coming
    from a learned model on the other side of a pipe.

    Attributes:
        :ivar list: stores info on the moves that have been performed during the game
        :ivar Config config: stores the whole config for how to run
        :ivar PlayConfig play_config: just stores the PlayConfig to use to play the game. Taken from the config
            if not specifically specified.
        :ivar int labels_n: length of self.labels.
        :ivar list(str) labels: all of the possible move labels (like a1b1, a1c1, etc...)
        :ivar dict(str,int) move_lookup: dict from move label to its index in self.labels
        :ivar list(Connection) pipe_pool: the pipes to send the observations of the game to to get back
            value and policy predictions from
        :ivar dict(str,Lock) node_lock: dict from FEN game state to a Lock, indicating
            whether that state is currently being explored by another thread.
        :ivar VisitStats tree: holds all of the visited game states and actions
            during the running of the AGZ algorithm
    """
    # dot = False

    def __init__(self, config: Config, pipes=None, play_config=None, dummy=False):
        self.moves = []

        self.tree = defaultdict(VisitStats)
        self.config = config
        self.play_config = play_config or self.config.play
        self.labels_n = config.n_labels
        self.labels = config.labels
        self.move_lookup = {chess.Move.from_uci(
            move): i for move, i in zip(self.labels, range(self.labels_n))}
        if dummy:
            return

        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)

    def reset(self):
        """
        reset the tree to begin a new exploration of states
        """
        self.tree = defaultdict(VisitStats)

    def action(self, env, can_stop=True) -> str:
        """
        Figures out the next best move
        within the specified environment and returns a string describing the action to take.

        :param ChessEnv env: environment in which to figure out the action
        :param boolean can_stop: whether we are allowed to take no action (return None)
        :return: None if no action should be taken (indicating a resign). Otherwise, returns a string
            indicating the action to take in uci format
        """
        self.reset()

        # for tl in range(self.play_config.thinking_loop):
        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(np.random.choice(range(self.labels_n),
                        p=self.apply_temperature(policy, env.num_halfmoves)))

        if can_stop and self.play_config.resign_threshold is not None and \
                root_value <= self.play_config.resign_threshold \
                and env.num_halfmoves > self.play_config.min_resign_turn:
            # noinspection PyTypeChecker
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[my_action]

    def search_moves(self, env):
        """
        Looks at all the possible moves using the AGZ MCTS algorithm
         and finds the highest value possible move. Does so using multiple threads to get multiple
         estimates from the AGZ MCTS algorithm so we can pick the best.

        :param ChessEnv env: env to search for moves within
        :return (float,float): the maximum value of all values predicted by each thread,
            and the first value that was predicted.
        """
        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(
                    self.search_my_move, env=env.copy(), is_root_node=True))

        vals = [f.result() for f in futures]

        return np.max(vals), vals[0]  # vals[0] is kind of racy

    def search_my_move(self, env: ChessEnv, is_root_node=False) -> float:
        """
        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)

        This method searches for possible moves, adds them to a search tree, and eventually returns the
        best move that was found during the search.

        :param ChessEnv env: environment in which to search for the move
        :param boolean is_root_node: whether this is the root node of the search.
        :return float: value of the move. This is calculated by getting a prediction
            from the value network.
        """
        if env.done:
            if env.winner == Winner.draw:
                return 0
            # assert env.whitewon != env.white_to_move # side to move can't be winner!
            return -1

        state = state_key(env)

        with self.node_lock[state]:
            if state not in self.tree:
                leaf_p, leaf_v = self.expand_and_evaluate(env)
                self.tree[state].p = leaf_p
                return leaf_v  # I'm returning everything from the POV of side to move

            # SELECT STEP
            action_t = self.select_action_q_and_u(env, is_root_node)

            virtual_loss = self.play_config.virtual_loss

            my_visit_stats = self.tree[state]
            my_stats = my_visit_stats.a[action_t]

            my_visit_stats.sum_n += virtual_loss
            my_stats.n += virtual_loss
            my_stats.w += -virtual_loss
            my_stats.q = my_stats.w / my_stats.n

        env.step(action_t.uci())
        leaf_v = self.search_my_move(env)  # next move from enemy POV
        leaf_v = -leaf_v

        # BACKUP STEP
        # on returning search path
        # update: N, W, Q
        with self.node_lock[state]:
            my_visit_stats.sum_n += -virtual_loss + 1
            my_stats.n += -virtual_loss + 1
            my_stats.w += virtual_loss + leaf_v
            my_stats.q = my_stats.w / my_stats.n

        return leaf_v

    def expand_and_evaluate(self, env):
        """ expand new leaf, this is called only once per state
        this is called with state locked
        insert P(a|s), return leaf_v

        This gets a prediction for the policy and value of the state within the given env
        :return (float, float): the policy and value predictions for this state
        """
        state_planes = env.canonical_input_planes()

        leaf_p, leaf_v = self.predict(state_planes)
        # these are canonical policy and value (i.e. side to move is "white")

        if not env.white_to_move:
            # get it back to python-chess form
            leaf_p = Config.flip_policy(leaf_p)

        return leaf_p, leaf_v

    def predict(self, state_planes):
        """
        Gets a prediction from the policy and value network
        :param state_planes: the observation state represented as planes
        :return (float,float): policy (prior probability of taking the action leading to this state)
            and value network (value of the state) prediction for this state.
        """
        pipe = self.pipe_pool.pop()
        pipe.send(state_planes)
        ret = pipe.recv()
        self.pipe_pool.append(pipe)
        return ret

    # @profile
    def select_action_q_and_u(self, env, is_root_node) -> chess.Move:
        """
        Picks the next action to explore using the AGZ MCTS algorithm.

        Picks based on the action which maximizes the maximum action value
        (ActionStats.q) + an upper confidence bound on that action.

        :param Environment env: env to look for the next moves within
        :param is_root_node: whether this is for the root node of the MCTS search.
        :return chess.Move: the move to explore
        """
        # this method is called with state locked
        state = state_key(env)

        my_visitstats = self.tree[state]

        if my_visitstats.p is not None:  # push p to edges
            tot_p = 1e-8
            for mov in env.board.legal_moves:
                mov_p = my_visitstats.p[self.move_lookup[mov]]
                my_visitstats.a[mov].p = mov_p
                tot_p += mov_p
            for a_s in my_visitstats.a.values():
                a_s.p /= tot_p
            my_visitstats.p = None

        # sqrt of sum(N(s, b); for all b)
        xx_ = np.sqrt(my_visitstats.sum_n + 1)

        e = self.play_config.noise_eps
        c_puct = self.play_config.c_puct
        dir_alpha = self.play_config.dirichlet_alpha

        best_s = -999
        best_a = None
        if is_root_node:
            noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))

        i = 0
        for action, a_s in my_visitstats.a.items():
            p_ = a_s.p
            if is_root_node:
                p_ = (1-e) * p_ + e * noise[i]
                i += 1
            b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
            if b > best_s:
                best_s = b
                best_a = action

        return best_a

    def apply_temperature(self, policy, turn):
        """
        Applies a random fluctuation to probability of choosing various actions
        :param policy: list of probabilities of taking each action
        :param turn: number of turns that have occurred in the game so far
        :return: policy, randomly perturbed based on the temperature. High temp = more perturbation. Low temp
            = less.
        """
        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1/tau)
            ret /= np.sum(ret)
            return ret

    def calc_policy(self, env):
        """calc π(a|s0)
        :return list(float): a list of probabilities of taking each action, calculated based on visit counts.
        """
        state = state_key(env)
        my_visitstats = self.tree[state]
        policy = np.zeros(self.labels_n)
        for action, a_s in my_visitstats.a.items():
            policy[self.move_lookup[action]] = a_s.n

        policy /= np.sum(policy)
        return policy

    def sl_action(self, observation, my_action, weight=1):
        """
        Logs the action in self.moves. Useful for generating a game using game data.

        :param str observation: FEN format observation indicating the game state
        :param str my_action: uci format action to take
        :param float weight: weight to assign to the taken action when logging it in self.moves
        :return str: the action, unmodified.
        """
        policy = np.zeros(self.labels_n)

        k = self.move_lookup[chess.Move.from_uci(my_action)]
        policy[k] = weight

        self.moves.append([observation, list(policy)])
        return my_action

    def finish_game(self, z):
        """
        When game is done, updates the value of all past moves based on the result.

        :param self:
        :param z: win=1, lose=-1, draw=0
        :return:
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]



def state_key(env: ChessEnv) -> str:
    """
    :param ChessEnv env: env to encode
    :return str: a str representation of the game state
    """
    fen = env.board.fen().rsplit(' ', 1)  # drop the move clock
    return fen[0]

from Config import Config
import numpy as np
import copy

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# input planes
# noinspection SpellCheckingInspection
pieces_order = 'KQRBNPkqrbnp'  # 12x8x8
castling_order = 'KQkq'       # 4x8x8
# fifty-move-rule             # 1x8x8
# en en_passant               # 1x8x8

ind = {pieces_order[i]: i for i in range(12)}




def find_pgn_files(directory, pattern='*.pgn'):
    dir_pattern = os.path.join(directory, pattern)
    files = list(sorted(glob(dir_pattern)))
    return files


def write_game_data_to_file(path, data):
    try:
        with open(path, "wt") as f:
            json.dump(data, f)
    except Exception as e:
        print(e)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    return SupervisedLearningWorker(config).start()


class SupervisedLearningWorker:
    """
    Worker which performs supervised learning on recorded games.

    Attributes:
        :ivar Config config: config for this worker
        :ivar list((str,list(float)) buffer: buffer containing the data to use for training -
            each entry contains a FEN encoded game state and a list where every index corresponds
            to a chess move. The move that was taken in the actual game is given a value (based on
            the player elo), all other moves are given a 0.
    """

    def __init__(self, config: Config):
        """
        :param config:
        """
        self.config = config
        self.buffer = []

    def start(self):
        """
        Start the actual training.
        """
        self.buffer = []
        # noinspection PyAttributeOutsideInit
        self.idx = 0
        start_time = time()
        with ProcessPoolExecutor(max_workers=5) as executor:
            games = self.get_games_from_all_files()
            # poisoned reference (memleak)
            for res in as_completed([executor.submit(get_buffer, self.config, game) for game in games]):
                self.idx += 1
                env, data = res.result()
                print(data)
                self.save_data(data)
                end_time = time()
                logger.debug(f"game {self.idx:4} time={(end_time - start_time):.3f}s "
                             f"halfmoves={env.num_halfmoves:3} {env.winner:12}"
                             f"{' by resign ' if env.resigned else '           '}"
                             f"{env.observation.split(' ')[0]}")
                start_time = end_time

        if len(self.buffer) > 0:
            self.flush_buffer()

    def get_games_from_all_files(self):
        """
        Loads game data from pgn files
        :return list(chess.pgn.Game): the games
        """
        files = find_pgn_files(self.config.resource.play_data_dir)
        print(files)
        games = []
        for filename in files:
            games.extend(get_games_from_file(filename))
        print("done reading")
        return games

    def save_data(self, data):
        """

        :param (str,list(float)) data: a FEN encoded game state and a list where every index corresponds
            to a chess move. The move that was taken in the actual game is given a value (based on
            the player elo), all other moves are given a 0.
        """
        self.buffer += data
        if self.idx % self.config.play_data.sl_nb_game_in_file == 0:
            self.flush_buffer()

    def flush_buffer(self):
        """
        Clears out the moves loaded into the buffer and saves the to file.
        """
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(
            rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file,
                        args=(path, self.buffer))
        thread.start()
        self.buffer = []


def get_games_from_file(filename):
    """

    :param str filename: file containing the pgn game data
    :return list(pgn.Game): chess games in that file
    """
    pgn = open(filename, errors='ignore')
    offsets = []
    while True:
        offset = pgn.tell()

        headers = chess.pgn.read_headers(pgn)
        if headers is None:
            break

        offsets.append(offset)
    n = len(offsets)

    print(f"found {n} games")
    games = []
    for offset in offsets:
        pgn.seek(offset)
        games.append(chess.pgn.read_game(pgn))
    return games


def clip_elo_policy(config, elo):
    # 0 until min_elo, 1 after max_elo, linear in between
    return min(1, max(0, elo - config.play_data.min_elo_policy) / (
        config.play_data.max_elo_policy - config.play_data.min_elo_policy))


def get_buffer(config, game):
    """
    Gets data to load into the buffer by playing a game using PGN data.
    :param Config config: config to use to play the game
    :param pgn.Game game: game to play
    :return list(str,list(float)): data from this game for the SupervisedLearningWorker.buffer
    """
    env = ChessEnv().reset()
    white = ChessPlayer(config, dummy=True)
    black = ChessPlayer(config, dummy=True)
    result = game.headers["Result"]
    white_elo, black_elo = int(game.headers["WhiteElo"]), int(
        game.headers["BlackElo"])
    white_weight = clip_elo_policy(config, white_elo)
    black_weight = clip_elo_policy(config, black_elo)

    actions = []
    while not game.is_end():
        game = game.variation(0)
        actions.append(game.move.uci())
    k = 0
    while not env.done and k < len(actions):
        if env.white_to_move:
            action = white.sl_action(
                env.observation, actions[k], weight=white_weight)  # ignore=True
        else:
            action = black.sl_action(
                env.observation, actions[k], weight=black_weight)  # ignore=True
        env.step(action, False)
        k += 1

    if not env.board.is_game_over() and result != '1/2-1/2':
        env.resigned = True
    if result == '1-0':
        env.winner = Winner.white
        black_win = -1
    elif result == '0-1':
        env.winner = Winner.black
        black_win = 1
    else:
        env.winner = Winner.draw
        black_win = 0

    black.finish_game(black_win)
    white.finish_game(-black_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])

    return env, data

if __name__ == '__main__':
    from Config import Config
    C = Config()
    start(C)