from GoGame.goclasses import GoBoard, BoardNode
import GoGame.uifunctions as ui
from typing import Tuple, List, Set
import GoGame.config as cf


def remove_dead(board: GoBoard) -> None:
    '''
    Waits for player input to mark or unmark dead stones.
    Clicking a live stone marks its whole connected group as dead (faded).
    Clicking an already-faded stone restores the whole group to life.
    Switches players at the end so both can mark dead stones.
    '''
    from GoGame.turn_options import remove_dead_turn_options
    board.killed_last_turn.clear()
    ui.update_scoring(board)

    while True:
        event, values = board.read_window()
        else_choice: bool = remove_dead_turn_options(board, event)
        if not else_choice:
            return

        row, col = values['-GRAPH-']
        found_piece, piece = board.find_piece_click([row, col])

        if not found_piece:
            continue

        if piece.stone_here_color == cf.rgb_grey:
            ui.def_popup("You can't mark empty areas as dead.", 2)
            continue

        # if stone is already marked dead (faded) — toggle it back to alive
        if piece.stone_here_color in (cf.rgb_peach, cf.rgb_lavender):
            _restore_group(board, piece)
            ui.refresh_board_pygame(board)
            continue

        # live stone — mark its whole connected group as dead (faded)
        _mark_group_dead(board, piece)
        ui.refresh_board_pygame(board)
        board.switch_player()
        return


def _mark_group_dead(board: GoBoard, piece: BoardNode) -> None:
    '''
    Uses flood fill to find all connected stones of the same color as piece,
    then fades them to peach (black stones) or lavender (white stones).
    Also stores original colors in board.dead_stone_log so they can be restored.
    '''
    from GoGame.scoringboard import flood_fill

    if not hasattr(board, 'dead_stone_log'):
        board.dead_stone_log = []

    series: Tuple[Set[BoardNode], Set[BoardNode]] = flood_fill(piece)
    piece_string: List[Tuple[Tuple[int, int], Tuple[int, int, int]]] = []

    for item in series[0]:
        original_color = item.stone_here_color
        piece_string.append(((item.row, item.col), original_color))
        if original_color == board.player_black.unicode:
            item.stone_here_color = cf.rgb_peach
        else:
            item.stone_here_color = cf.rgb_lavender

    board.dead_stone_log.append(piece_string)


def restore_all_dead(board: GoBoard) -> None:
    '''Restores all faded dead stones to their original colors, clears dead_stone_log,
    and resets whose_turn to what it was when scoring began.'''
    if not hasattr(board, 'dead_stone_log'):
        return
    for piece_string in board.dead_stone_log:
        for (row, col), original_color in piece_string:
            board.board[row][col].stone_here_color = original_color
    board.dead_stone_log = []
    # restore the correct player turn
    if hasattr(board, 'scoring_start_turn'):
        if board.scoring_start_turn == board.player_black:
            board.whose_turn = board.player_black
            board.not_whose_turn = board.player_white
        else:
            board.whose_turn = board.player_white
            board.not_whose_turn = board.player_black
    ui.refresh_board_pygame(board)


def _restore_group(board: GoBoard, piece: BoardNode) -> None:
    '''
    Finds the group in dead_stone_log that contains this piece and
    restores all stones in that group to their original colors.
    '''
    if not hasattr(board, 'dead_stone_log'):
        return

    # find which log entry contains this piece
    for i, piece_string in enumerate(board.dead_stone_log):
        locations = [tpl[0] for tpl in piece_string]
        if (piece.row, piece.col) in locations:
            # restore all stones in this group
            for (row, col), original_color in piece_string:
                board.board[row][col].stone_here_color = original_color
            board.dead_stone_log.pop(i)
            return


def finalize_dead_stones(board: GoBoard) -> None:
    '''
    Called when scoring is finalized (both players pass during scoring).
    Converts all faded (peach/lavender) stones to grey and updates scores.
    '''
    if not hasattr(board, 'dead_stone_log'):
        return

    for piece_string in board.dead_stone_log:
        temp_list = []
        for (row, col), original_color in piece_string:
            board.board[row][col].stone_here_color = cf.rgb_grey
            temp_list.append((original_color, row, col, "Scoring"))
        board.killed_log.append(temp_list)
        board.position_played_log.append("Dead Removed")
        board.turn_num += 1

    board.dead_stone_log = []
    ui.refresh_board_pygame(board)


def remove_dead_undo_list(board: GoBoard,
                           piece_string: List[Tuple[Tuple[int, int],
                           Tuple[int, int, int]]]) -> None:
    '''Restores a piece_string to original colors (kept for compatibility).'''
    for tpl in piece_string:
        item: BoardNode = board.board[tpl[0][0]][tpl[0][1]]
        if item.stone_here_color == cf.rgb_peach:
            item.stone_here_color = board.player_black.unicode
        elif item.stone_here_color == cf.rgb_lavender:
            item.stone_here_color = board.player_white.unicode
    ui.refresh_board_pygame(board)


def remove_stones_and_update_score(board: GoBoard,
                                    piece_string: List[Tuple[Tuple[int, int],
                                    Tuple[int, int, int]]]) -> None:
    '''Kept for compatibility — finalize_dead_stones() replaces this in scoring.'''
    for tpl in piece_string:
        item: BoardNode = board.board[tpl[0][0]][tpl[0][1]]
        item.stone_here_color = cf.rgb_grey
    ui.refresh_board_pygame(board)
    temp_list = []
    for item in piece_string:
        temp_list.append((item[1], item[0][0], item[0][1], "Scoring"))
    board.killed_log.append(temp_list)
    board.position_played_log.append("Dead Removed")
    board.turn_num += 1
