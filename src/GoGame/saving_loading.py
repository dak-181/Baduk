import pygame
from typing import Union
from GoGame.goclasses import GoBoard
import GoGame.pygame_ui as pg_ui


def move_to_pkl_directory() -> str:
    "Returns the absolute path to the pklfiles subdirectory, creating it if needed. Does NOT chdir."
    from os import getcwd, path, makedirs
    import re
    wd = getcwd()
    # strip any trailing pklfiles segments to always resolve from project root
    base = re.sub(r'[/\\]pklfiles$', '', wd)
    full_path = path.join(base, 'pklfiles')
    makedirs(full_path, exist_ok=True)
    return full_path


def save_pickle(board: GoBoard) -> None:
    '''Saves the game to a .pkl file. Does not allow overwriting.'''
    import pickle
    filename = save_pickle_name_choice()
    if not filename:
        return

    # temporarily remove unpicklable pygame objects
    backup_screen       = board.screen
    backup_backup_board = getattr(board, 'backup_board', None)
    backup_btn_rects    = getattr(board, 'btn_rects', None)
    board.screen        = None
    board.backup_board  = None
    board.btn_rects     = None
    if hasattr(board, 'scoring_dead'):
        del board.scoring_dead

    with open(filename, "wb") as pkl_file:
        pickle.dump(board, pkl_file)

    board.screen       = backup_screen
    board.backup_board = backup_backup_board
    board.btn_rects    = backup_btn_rects


def save_pickle_name_choice() -> Union[None, str]:
    """Ask the user for a filename; return full path or None on cancel."""
    from os import path
    from GoGame.uifunctions import default_popup_no_button
    full_path = move_to_pkl_directory()

    while True:
        text = "Enter a save-file name (no extension):"
        filename = pg_ui.popup_get_text(text, title="Save Game",
                                        ok_label="Save", cancel_label="Don't Save")
        if filename is None:
            return None
        filename = filename.strip()
        if not filename:
            continue
        full = path.join(full_path, f"{filename}.pkl")
        if path.isfile(full):
            default_popup_no_button("That file already exists. Choose a different name.", 2)
            import time; time.sleep(2)
            continue
        return full


def load_pkl(input_path: str) -> GoBoard:
    '''Loads a game from a .pkl file.'''
    import pickle
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def choose_file() -> None:
    "Let the user pick a .pkl save file and resume that game."
    from GoGame.uifunctions import setup_board_window_pygame, default_popup_no_button
    pkl_dir = move_to_pkl_directory()

    file_path = pg_ui.popup_get_file("Select a save file", title="Load Game",
                                     initial_folder=pkl_dir)
    if not file_path:
        return

    import os
    fname = os.path.basename(file_path)
    default_popup_no_button(f"Loading: {fname}", 2)

    friend = load_pkl(file_path)
    setup_board_window_pygame(friend)
    friend.play_game(from_file=True, fixes_handicap=False)
