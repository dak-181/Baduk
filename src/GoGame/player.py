import GoGame.uifunctions as ui
import GoGame.pygame_ui as pg_ui
from typing import Optional, Tuple, Type


class Player():
    def __init__(self, name: Optional[str] = None, color: Optional[str] = None,
                 komi: float = 0, unicode_choice: Optional[tuple] = None):
        self.name: Optional[str] = name
        self.color: Optional[str] = color
        self.komi: float = komi
        self.unicode: Tuple[int, int, int] = unicode_choice
        self.territory: int = 0
        self.black_set_len: int = 0
        self.white_set_len: int = 0

    @staticmethod
    def setup_player(defaults, nme, clr, uc) -> Type['Player']:
        if defaults:
            if clr == "Black":
                player_assignment = Player(name=nme, color=clr, unicode_choice=uc)
            else:
                player_assignment = Player(name=nme, color=clr, komi=7.5, unicode_choice=uc)
        else:
            player_assignment = Player(color=clr, unicode_choice=uc)
            player_assignment.choose_name()
            player_assignment.choose_komi()
        return player_assignment

    @staticmethod
    def get_input(info, conversion_func):
        '''Gets user input via pygame dialog, with error handling.'''
        done = False
        while not done:
            try:
                user_input = ui.validation_gui(info, conversion_func)
                done = True
            except ValueError:
                ui.default_popup_no_button("Invalid input. Please try again", 2)
        return user_input

    def choose_name(self) -> None:
        "Allows a player to choose their name."
        answer = pg_ui.popup_yes_no(
            "Would you like to change your name?",
            title="Player Name"
        )
        if answer == "No":
            self.name = "Player Two" if self.color == "White" else "Player One"
            return
        self.name = Player.get_input(
            "Enter a name (max 30 characters):",
            lambda x: str(x)[:30]
        )

    def choose_komi(self) -> None:
        "Allows a player to choose their komi."
        answer = pg_ui.popup_yes_no(
            "Would you like to change your Komi?",
            title="Komi"
        )
        if answer == "No":
            if self.color == "White":
                self.komi = 7.5
            return
        self.komi = Player.get_input(
            f"Your color is {self.color}. Enter Komi (7.5 is standard for White):",
            float
        )
