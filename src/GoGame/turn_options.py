import GoGame.uifunctions as ui
import GoGame.pygame_ui as pg_ui
from GoGame.goclasses import GoBoard
from typing import Optional

# sentinel that matches the old sg.WIN_CLOSED usage
WIN_CLOSED = None


def normal_turn_options(board: GoBoard, event, text: Optional[str] = None) -> None:
    '''
    Handles various game options based on the given event.
    Options: Pass Turn, Quit Program, WIN_CLOSED, Save Game, Undo Turn, Exit To Menu.
    In scoring mode "Quit Program" is shown as "Resume Game" in the UI,
    but the event value passed here is still "Quit Program".
    '''
    if event in (WIN_CLOSED, "Quit Program"):
        import pygame, sys
        pygame.quit()
        sys.exit()

    if event == "Pass Turn":
        ui.def_popup("Skipped turn", 1)
        board.times_passed += 1
        board.turn_num += 1
        board.position_played_log.append((text, -3, -3))
        board.killed_log.append([])
        board.switch_player()

    elif event == "Save Game":
        from GoGame.saving_loading import save_pickle
        save_pickle(board)

    elif event == "Undo Turn":
        if board.turn_num == 0:
            ui.def_popup("You can't undo when nothing has happened.", 2)
        else:
            from GoGame.undoing import undo_checker
            undo_checker(board)
            return

    elif event == "Exit To Menu":
        from GoGame.main import play_game_main
        ui.close_window(board)
        play_game_main()
        import sys; sys.exit()

    else:
        raise ValueError(f"Unhandled event: {event!r}")


def remove_dead_turn_options(board: GoBoard, event) -> bool:
    '''
    Handles events during the dead-stone removal / scoring phase.
    Returns True only if the player clicked a board intersection.
    '''
    if event == "Pass Turn":
        normal_turn_options(board, event, text="Scoring Passed")
        return False
    elif event == "Save Game":
        from GoGame.saving_loading import save_pickle
        save_pickle(board)
        return False
    elif event == "Quit Program":
        # In scoring mode this button is labelled "Resume Game" in the UI
        board.mode = "Playing"
        board.mode_change = True
        board.resuming_scoring_buffer("Resumed")
        return False
    elif event == "Undo Turn":
        normal_turn_options(board, event)
        return False
    elif event == "Exit To Menu":
        normal_turn_options(board, event)
        return False
    elif event == WIN_CLOSED:
        import pygame, sys
        pygame.quit(); sys.exit()
    else:
        return True
