from GoGame.handicap import Handicap
from GoGame.goclasses import GoBoard, BoardNode, BoardString
import sys
from typing import Tuple, Optional, List, Set, Union, Type
from GoGame.scoringboard import ScoringBoard
from GoGame.goclasses import play_turn_bot_helper
import GoGame.config as cf
import tensorflow as tf

sys.setrecursionlimit(10000)

# enable GPU memory growth — prevents TF grabbing all VRAM at startup
# Guard against RuntimeError when TF context is already initialized (e.g. when
# loading a saved game causes a second import after TF has already been used).
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU enabled: {[g.name for g in gpus]}")
    except RuntimeError:
        # Context already initialized — memory growth must be set before first use.
        # This is harmless; TF will use whatever allocation strategy was set first.
        pass

# safety cap — 19x19 games can run longer than 9x9
MAX_TURNS = 600


def initializing_game(nn, nn_bad, board_size: int = cf.AI_BOARD_SIZE,
                       defaults: Optional[bool] = True,
                       state: Optional[dict] = None):
    """
    Initialize and run one AI self-play game.
    Returns True if black won, False if white won.
    """
    game_board = NNBoard(nn, nn_bad, board_size, defaults, state=state)
    return game_board.play_game(False)


class NNBoard(GoBoard):
    def __init__(self, nn, nn_bad, board_size=9, defaults=True,
                 state: Optional[dict] = None):
        # store nn/nn_bad and state before super().__init__
        # so any overridden methods called during init can access them
        self.nn     = nn
        self.nn_bad = nn_bad
        self._state = state
        self.ai_training_info: List[str] = []
        self.ai_output_info:   List[List[float]] = []

        super().__init__(board_size, defaults)

        # override pygame attributes — no window for headless self-play
        self.window            = None
        self.screen            = None
        self.backup_board      = None
        self.pygame_board_vals = None
        self.btn_rects         = {}

    # ── cancellation helper ───────────────────────────────────────────────
    def _cancelled(self) -> bool:
        return bool(self._state and self._state.get("cancelled"))

    # ── play_game ─────────────────────────────────────────────────────────
    def play_game(self, from_file: Optional[bool] = False,
                  fixes_handicap: Optional[bool] = False):
        empty_board = '0' * (self.board_size * self.board_size)
        for _ in range(10):
            self.ai_training_info.append(empty_board)
        self.ai_white_board = empty_board
        self.ai_black_board = '1' * (self.board_size * self.board_size)
        self.resigned_player = None  # set to the resigning Player if game ends by resignation
        self._consecutive_low_value = 0  # counts consecutive turns below resignation threshold
        return self.play_game_playing_mode(from_file, fixes_handicap)

    def play_game_playing_mode(self, from_file, fixes_handicap) -> bool:
        if not from_file:
            self.board = self.setup_board()
        if fixes_handicap:
            hc = Handicap(self)
            self.handicap = hc.custom_handicap(False)
        self.turn_loop()
        self.mode = "Scoring"
        self.times_passed = 0
        self.resuming_scoring_buffer("Scoring")
        return self.playing_mode_end_of_game()

    def playing_mode_end_of_game(self) -> bool:
        """Score the game and save training data to saved_self_play.jsonl."""
        # If the game ended by resignation, the resigning player lost.
        # Skip territory scoring entirely — the board is unfinished and dead
        # stones haven't been removed, so the score would be unreliable.
        if getattr(self, 'resigned_player', None) is not None:
            winner = 0 if self.resigned_player == self.player_black else 1
            print(f"winner is {winner} (by resignation)")
        else:
            winner = self.making_score_board_object()
            print(f"winner is {winner}")

        save_selfplay_sgf(self, winner)
        import json

        def perspective_value(winner: int, turn_color: int) -> float:
            """Return +1.0 if the player to move won, -1.0 if they lost.
            winner:     1 = black won, 0 = white won
            turn_color: 1 = black to move, 2 = white to move
            """
            black_won = (winner == 1)
            black_to_move = (turn_color == 1)
            return 1.0 if (black_won == black_to_move) else -1.0

        # Stream-append one line per sample — never loads the full file into memory
        with open('saved_self_play.jsonl', 'a') as fn2:
            for item in self.ai_output_info:
                sample = (item[0], item[1], perspective_value(int(winner), item[2]))
                fn2.write(json.dumps(sample) + '\n')

        return winner

    # ── turn loop with exit conditions ────────────────────────────────────
    def turn_loop(self) -> None:
        """
        Plays turns until both players pass consecutively, the game exceeds
        MAX_TURNS, or a cancellation is requested.
        """
        while self.times_passed <= 1:
            if self._cancelled():
                print("Self-play cancelled — forcing game end.")
                self.times_passed = 2
                break
            if self.turn_num >= MAX_TURNS:
                print(f"MAX_TURNS ({MAX_TURNS}) reached — forcing passes to end game.")
                self.times_passed = 2
                break
            self.play_turn(good_bot=True)

    def play_turn(self, good_bot: bool = False) -> None:
        """Plays one AI turn using NNMCST. Has a placement-attempt limit."""
        from GoGame.nnmcst import NNMCST
        import copy, time

        MAX_PLACEMENT_ATTEMPTS = 10
        attempt = 0
        truth_value: bool = False

        while not truth_value:
            if self._cancelled():
                return
            if attempt >= MAX_PLACEMENT_ATTEMPTS:
                # Force a pass rather than looping forever
                print(f"  Placement failed {attempt} times — forcing pass.")
                self.times_passed += 1
                self.turn_num += 1
                self.position_played_log.append(("Pass", -3, -3))
                self.killed_log.append([])
                self.preprevious_board_state = self.previous_board_state
                self.previous_board_state = self.make_board_string()[1:]
                self.switch_player()
                return

            self.board_copy: List[BoardString] = copy.deepcopy(self.board)
            t0 = time.time()

            chosen_nn = self.nn if good_bot else self.nn_bad
            self.turn_nnmcst = NNMCST(
                self.board_copy, self.ai_training_info,
                self.ai_white_board, self.ai_black_board,
                cf.MCTS_ITERATIONS, (self.whose_turn, self.not_whose_turn),
                chosen_nn, self.turn_num,
                training=True
            )
            val, output_chances, formatted_ai_training_info, root_value = self.turn_nnmcst.run_mcst()

            truth_value = play_turn_bot_helper(self, truth_value, val)
            if truth_value == "Break" or truth_value == "Passed":
                if truth_value == "Passed":
                    self.preprevious_board_state = self.previous_board_state
                    self.previous_board_state = self.make_board_string()[1:]
                return
            if not truth_value:
                attempt += 1
                continue

            # move was accepted — record training sample for this turn
            turn_color = 1 if self.whose_turn == self.player_black else 2
            self.ai_output_info.append((formatted_ai_training_info, output_chances, turn_color))

            # print board state and timing only on the successful turn
            t1 = time.time()
            self.print_board()
            print(f"  t={t1-t0:.2f}s  val={val}  pos={val//self.board_size},{val%self.board_size}  turn={self.turn_num}")

            # resignation: if the current player's position is hopeless for two
            # consecutive turns and we're past the opening, end the game.
            # Threshold: -0.95. Minimum turn: 50. Can be disabled via config.
            if cf.ALLOW_RESIGNATION and self.turn_num > 50 and root_value < cf.RESIGNATION_THRESHOLD:
                self._consecutive_low_value += 1
                if self._consecutive_low_value >= 2:
                    print(f"  Resigned at turn {self.turn_num} (value={root_value:.3f})")
                    self.resigned_player = self.not_whose_turn
                    self.times_passed = 2
            else:
                self._consecutive_low_value = 0

        self.make_turn_info()

    def make_turn_info(self) -> None:
        temp_list: List[Tuple[Tuple[int, int, int], int, int]] = []
        for item in self.killed_last_turn:
            temp_list.append((self.not_whose_turn.unicode, item.row, item.col))
        self.killed_log.append(temp_list)
        self.ai_training_info.append(self.make_board_string())
        self.preprevious_board_state = self.previous_board_state
        self.previous_board_state = self.make_board_string()[1:]
        self.switch_player()

    def print_board(self) -> None:
        """Prints the current board state as emoji, read from self.board.
        Skips if no stones have been placed yet."""
        has_stone = any(
            node.stone_here_color != cf.rgb_grey
            for row in self.board for node in row
        )
        if not has_stone:
            return
        for row in self.board:
            line = ""
            for node in row:
                if node.stone_here_color == cf.rgb_black:
                    line += '\u26AB'
                elif node.stone_here_color == cf.rgb_white:
                    line += '\u26AA'
                else:
                    line += '\u26D4'
            print(line)

    def making_score_board_object(self):
        self.scoring_dead = NNScoringBoard(self)
        return self.scoring_dead.dealing_with_dead_stones()


def save_selfplay_sgf(board: 'NNBoard', winner: int) -> None:
    """
    Write a standard SGF file for a completed self-play game to traininghistory/.
    Filename: traininghistory/game_NNNN_<result>.sgf
    Picks up from the highest existing file number so runs never overwrite.
    """
    import os, re

    sgf_dir = 'traininghistory'
    os.makedirs(sgf_dir, exist_ok=True)

    # derive next game number from existing files
    existing = [f for f in os.listdir(sgf_dir) if f.endswith('.sgf')]
    game_num = 1
    for fname in existing:
        m = re.match(r'game_(\d+)', fname)
        if m:
            game_num = max(game_num, int(m.group(1)) + 1)

    # build RE (result) field
    resigned = getattr(board, 'resigned_player', None)
    if resigned is not None:
        result_label = 'Black_wins_by_resignation' if winner == 1 else 'White_wins_by_resignation'
        re_str = 'B+R' if winner == 1 else 'W+R'
    else:
        pb = board.player_black
        pw = board.player_white
        b_total = pb.komi + pb.territory + pb.black_set_len
        w_total = pw.komi + pw.territory + pw.white_set_len
        diff = abs(b_total - w_total)
        winner_label = 'Black' if winner == 1 else 'White'
        result_label = f'{winner_label}_wins_by_score_B{b_total}_W{w_total}'
        re_str = f'B+{diff:.1f}' if winner == 1 else f'W+{diff:.1f}'

    filename = os.path.join(sgf_dir, f'game_{game_num:04d}_{result_label}.sgf')

    # SGF coordinate helper: (row, col) -> two-char string e.g. (3,15) -> 'pd'
    def to_sgf_coord(row: int, col: int) -> str:
        return chr(ord('a') + col) + chr(ord('a') + row)

    # build move sequence from position_played_log
    # entries are: (color, row, col), ("Pass", -3, -3), or plain strings
    # track whose turn it is by alternating — black always moves first
    move_tokens = []
    current_color = 'B'
    for entry in board.position_played_log:
        if not isinstance(entry, tuple):
            continue  # skip "Scoring", "Resumed", "Dead Removed" etc.
        color, row, col = entry
        if color == 'Black':
            move_tokens.append(('B', to_sgf_coord(row, col)))
            current_color = 'W'
        elif color == 'White':
            move_tokens.append(('W', to_sgf_coord(row, col)))
            current_color = 'B'
        elif color == 'Pass':
            # SGF pass is an empty coordinate []
            move_tokens.append((current_color, ''))
            current_color = 'W' if current_color == 'B' else 'B' 

    komi = board.player_white.komi
    size = board.board_size

    lines = [f'(;GM[1]FF[4]CA[UTF-8]SZ[{size}]KM[{komi}]RE[{re_str}]PB[AI]PW[AI]']
    for color, coord in move_tokens:
        lines.append(f';{color}[{coord}]')
    lines.append(')')

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"  [sgf] saved: {filename}")
    except Exception as e:
        print(f"  [sgf] failed: {e}")


class NNScoringBoard(ScoringBoard):
    def __init__(self, parent_obj: Type[GoBoard]) -> None:
        self.parent     = parent_obj
        self.defaults   = self.parent.defaults
        self.board_size = self.parent.board_size
        self.board      = self.parent.board

        self.init_helper_player_deep()

        self.times_passed        = self.parent.times_passed
        self.turn_num            = self.parent.turn_num
        self.position_played_log = self.parent.position_played_log
        self.visit_kill          = self.parent.visit_kill
        self.killed_last_turn    = self.parent.killed_last_turn
        self.killed_log          = self.parent.killed_log

        self.mode        = self.parent.mode
        self.mode_change = self.parent.mode_change
        self.handicap    = self.parent.handicap

        self.empty_strings: List[BoardString] = []
        self.black_strings: List[BoardString] = []
        self.white_strings: List[BoardString] = []
