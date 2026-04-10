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
