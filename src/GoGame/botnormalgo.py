import GoGame.uifunctions as ui
from GoGame.goclasses import GoBoard
from random import randrange
from GoGame.goclasses import play_turn_bot_helper
from typing import Union, Literal, Optional
from GoGame.handicap import Handicap
import time


class BotBoard(GoBoard):
    def __init__(self, board_size=9, defaults=True):
        super().__init__(board_size, defaults)

    def playing_mode_end_of_game(self) -> bool:
        "Generates a score_board object and does automatic scoring. Returns True if black won."
        from GoGame.scoringboard import making_score_board_object
        winner = making_score_board_object(self)
        return winner

    def play_game_playing_mode(self, from_file, fixes_handicap) -> bool:
        '''
        Overrides GoBoard.play_game_playing_mode so the bot's turn_loop is used
        instead of the human play_turn loop.
        '''
        if not from_file:
            self.board = self.setup_board()
        else:
            ui.refresh_board_pygame(self)
        if fixes_handicap:
            from GoGame.handicap import Handicap
            hc = Handicap(self)
            self.handicap = hc.custom_handicap(False)
        self.turn_loop()
        self.mode = "Scoring"
        self.times_passed = 0
        self.resuming_scoring_buffer("Scoring")
        return self.playing_mode_end_of_game()

    def turn_loop(self) -> None:
        "Plays turns in a loop: human plays black, bot plays white."
        while self.times_passed <= 1:
            if self.whose_turn == self.player_black:
                self.play_turn()
            else:
                self.play_turn_bot()

    def play_turn_bot(self) -> None:
        "Generates a random location for the bot to play, then plays the turn."
        ui.update_scoring(self)
        truth_value: Union[bool, Literal['Passed']] = False
        tries = 0
        while not truth_value:
            val = randrange(0, (self.board_size * self.board_size))
            tries += 1
            if tries >= 120:
                val = self.board_size * self.board_size
            truth_value = play_turn_bot_helper(self, truth_value, val)
            time.sleep(.5)
            ui.refresh_board_pygame(self)
            if truth_value == "Passed":
                return
        self.make_turn_info()


class NNBotBoard(GoBoard):
    """Human (black) vs neural-net AI (white). Loads weights from a given .h5 file."""

    def __init__(self, board_size: int = 19, defaults: bool = True,
                 weights_path: str = "other_play.weights.h5"):
        super().__init__(board_size, defaults)
        self.weights_path = weights_path
        self._nn_model = None   # lazy-loaded on first bot turn

    def _get_nn(self):
        if self._nn_model is None:
            from GoGame.neuralnet import nn_model_from_file
            self._nn_model = nn_model_from_file(self.weights_path, self.board_size)
        return self._nn_model

    def playing_mode_end_of_game(self) -> bool:
        ui.scoring_mode_popup()
        return self.scoring_block()

    def play_game_playing_mode(self, from_file, fixes_handicap) -> bool:
        if not from_file:
            self.board = self.setup_board()
        else:
            ui.refresh_board_pygame(self)
        if fixes_handicap:
            hc = Handicap(self)
            self.handicap = hc.custom_handicap(False)
        self.turn_loop()
        self.mode = "Scoring"
        self.times_passed = 0
        self.resuming_scoring_buffer("Scoring")
        return self.playing_mode_end_of_game()

    def turn_loop(self) -> None:
        while self.times_passed <= 1:
            if self.whose_turn == self.player_black:
                self.play_turn()
            else:
                self.play_turn_nn()

    def play_turn_nn(self) -> None:
        """Uses NNMCST to pick white's move, mirrors NNBoard.play_turn logic."""
        import copy
        from GoGame.nnmcst import NNMCST
        import GoGame.config as cf

        ui.update_scoring(self)

        # seed training-info history if not already present
        if not hasattr(self, 'ai_training_info'):
            empty = '0' * (self.board_size * self.board_size)
            self.ai_training_info = ['0' + empty] * 10
            self.ai_white_board   = empty
            self.ai_black_board   = '1' * (self.board_size * self.board_size)

        MAX_ATTEMPTS = 10
        MAX_FIRSTLINE_RETRIES = 5
        attempt = 0
        truth_value: Union[bool, Literal['Passed', 'Break']] = False

        # build first-line index set for filtering early moves
        _bs = self.board_size
        _first_line = set()
        for i in range(_bs):
            _first_line.add(i)                   # row 0
            _first_line.add((_bs - 1) * _bs + i) # row 18
            _first_line.add(i * _bs)              # col 0
            _first_line.add(i * _bs + (_bs - 1)) # col 18

        while not truth_value:
            if attempt >= MAX_ATTEMPTS:
                # force a pass
                self.times_passed += 1
                self.turn_num += 1
                self.position_played_log.append(("Pass", -3, -3))
                self.killed_log.append([])
                self.switch_player()
                ui.def_popup("AI passed.", 2)
                ui.refresh_board_pygame(self)
                return

            board_copy = copy.deepcopy(self.board)
            mcts = NNMCST(
                board_copy,
                self.ai_training_info,
                self.ai_black_board,
                self.ai_white_board,
                cf.PLAY_MCTS_ITERATIONS,
                (self.whose_turn, self.not_whose_turn),
                self._get_nn(),
                self.turn_num,
            )
            val, _policy, formatted_info = mcts.run_mcst()

            # re-run MCTS if it picks a first-line move in the first 100 turns
            if self.turn_num < 100 and val in _first_line:
                firstline_retries = 0
                while val in _first_line and firstline_retries < MAX_FIRSTLINE_RETRIES:
                    board_copy = copy.deepcopy(self.board)
                    mcts = NNMCST(
                        board_copy,
                        self.ai_training_info,
                        self.ai_black_board,
                        self.ai_white_board,
                        cf.PLAY_MCTS_ITERATIONS,
                        (self.whose_turn, self.not_whose_turn),
                        self._get_nn(),
                        self.turn_num,
                    )
                    val, _policy, formatted_info = mcts.run_mcst()
                    firstline_retries += 1

            truth_value = play_turn_bot_helper(self, truth_value, val)
            if truth_value in ("Break", "Passed"):
                if truth_value == "Passed":
                    ui.def_popup("AI passed.", 2)
                ui.refresh_board_pygame(self)
                return
            if not truth_value:
                attempt += 1

        # record new board state in history
        self.ai_training_info.append(self.make_board_string())
        self._make_turn_info_nn()
        ui.refresh_board_pygame(self)

    def _make_turn_info_nn(self) -> None:
        temp = []
        for item in self.killed_last_turn:
            temp.append((self.not_whose_turn.unicode, item.row, item.col))
        self.killed_log.append(temp)
        self.switch_player()
