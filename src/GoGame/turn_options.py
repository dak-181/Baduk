import GoGame.uifunctions as ui
import GoGame.pygame_ui as pg_ui
from GoGame.goclasses import GoBoard
from typing import Optional

# sentinel that matches the old sg.WIN_CLOSED usage
WIN_CLOSED = None


def _is_ai_board(board: GoBoard) -> bool:
    """Returns True if the board is a human vs AI game (BotBoard or NNBotBoard)."""
    return hasattr(board, '_nn_model') or hasattr(board, 'play_turn_bot')


def normal_turn_options(board: GoBoard, event, text: Optional[str] = None) -> None:
    '''
    Handles various game options based on the given event.
    Options: Pass Turn, Save Game, Undo Turn, Resume Game, Exit To Menu.
    '''
    if event in (WIN_CLOSED, "Quit Program"):
        import pygame, sys
        pygame.quit()
        sys.exit()

    if event == "Pass Turn":
        ui.def_popup("Skipped turn", 2)
        board.times_passed += 1
        board.turn_num += 1
        board.position_played_log.append((text, -3, -3))
        board.killed_log.append([])
        board.preprevious_board_state = board.previous_board_state
        board.previous_board_state = board.make_board_string()[1:]
        board.switch_player()

    elif event == "Save Game":
        from GoGame.saving_loading import save_pickle
        save_pickle(board)

    elif event == "Undo Turn":
        if board.turn_num == 0:
            ui.def_popup("You can't undo when nothing has happened.", 2)
        elif _is_ai_board(board) and board.whose_turn == board.player_black:
            # It's the human's turn but the last move was the AI's — block undo
            # so the player can't cherry-pick AI responses.
            ui.def_popup("You can't undo during a game against the AI.", 2)
        else:
            from GoGame.undoing import undo_checker
            undo_checker(board)
            return

    elif event == "Exit To Menu":
        from GoGame.main import play_game_main
        ui.close_window(board)
        play_game_main()
        import sys; sys.exit()

    elif event == "Resume Game":
        ui.def_popup("Game is not paused, you can't resume a game that's not in scoring.", 2)

    else:
        raise ValueError(f"Unhandled event: {event!r}")


def remove_dead_turn_options(board: GoBoard, event) -> bool:
    '''
    Handles events during the dead-stone removal / scoring phase.
    Returns True only if the player clicked a board intersection.
    '''
    if event in ("Pass Turn", "Accept"):
        board.times_passed += 1
        board.turn_num += 1
        board.position_played_log.append(("Scoring Passed", -3, -3))
        board.killed_log.append([])
        board.switch_player()
        return False
    elif event == "Save Game":
        from GoGame.saving_loading import save_pickle
        save_pickle(board)
        return False
    elif event == "Resume Game":
        # restore any faded dead stones before resuming play
        from GoGame.remove_dead import restore_all_dead
        restore_all_dead(board)
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
