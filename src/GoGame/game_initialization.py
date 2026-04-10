import GoGame.uifunctions as ui
import GoGame.pygame_ui as pg_ui
from typing import Optional


def initializing_game(board_size: int, defaults: bool = True,
                      vs_bot: bool = False) -> None:
    '''
    Initialize a new game based on user preferences.
    Parameters:
        board_size: The size of the game board.
        defaults: If True, use default settings;
            otherwise, allow the user to modify player names and komi.
        vs_bot: If True, play against an AI opponent.
    '''
    game_board = initialize_player_choice(board_size, defaults, vs_bot)
    ui.setup_board_window_pygame(game_board)
    handicap_info = initialize_handicap_choice(defaults)
    game_board.play_game(fixes_handicap=handicap_info)


def initializing_game_vs_nn(weights_path: str, defaults: bool = True) -> None:
    """Start a human vs neural-net game using the given .h5 weights file."""
    import GoGame.config as cf
    from GoGame.botnormalgo import NNBotBoard
    board_size = cf.AI_BOARD_SIZE
    game_board = NNBotBoard(board_size, defaults, weights_path=weights_path)
    ui.setup_board_window_pygame(game_board)
    game_board.play_game(fixes_handicap=False)


def choose_board_type(vs_bot: bool = False, board_size: int = 9, defaults: bool = True):
    '''
    Choose the correct type of board (GoBoard or BotBoard).
    '''
    if vs_bot:
        from GoGame.botnormalgo import BotBoard
        return BotBoard(board_size, defaults)
    else:
        from GoGame.goclasses import GoBoard
        return GoBoard(board_size, defaults)


def initialize_handicap_choice(defaults: Optional[bool]) -> bool:
    'Requests info from player regarding handicap. Returns a bool.'
    if not defaults:
        if request_handicap_info():
            return True
    return False


def request_handicap_info() -> bool:
    "Asks the player if they want a handicap. Returns True for Yes."
    answer = pg_ui.popup_yes_no(
        "Would you like to modify the handicap?",
        title="Handicap"
    )
    return answer == "Yes"


def initialize_player_choice(board_size: int, defaults: bool = True,
                              vs_bot: bool = False):
    "Asks the player if they wish to change their name or komi. Returns a GoBoard or BotBoard."
    if not defaults:
        answer = pg_ui.popup_yes_no(
            "Would you like to modify player names and komi?",
            title="Player Setup"
        )
        if answer == "No":
            defaults = True
    return choose_board_type(vs_bot, board_size, defaults)
