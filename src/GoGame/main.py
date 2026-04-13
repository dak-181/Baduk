import pygame
import multiprocessing
import GoGame.game_initialization as start
import GoGame.uifunctions as ui


def play_game_main():
    '''
    Main menu loop. Uses ui.start_game_menu() which returns the chosen
    action as a string.
    '''
    while True:
        event = ui.start_game_menu()

        if event == "Load Game":
            from GoGame.saving_loading import choose_file
            choose_file()
            continue

        elif event == "New Custom Game":
            board_size = ui.start_game()
            start.initializing_game(board_size, defaults=False)

        elif event == "New Default Game":
            start.initializing_game(9, True)

        elif event == "Play Against AI":
            _play_against_ai()

        elif event == "AI SelfPlay":
            _run_self_play()

        elif event == "AI Training":
            _run_ai_training()

        elif event == "Import SGF Files":
            _run_import_sgf()

        elif event == "SGF Training":
            _run_sgf_training()

        elif event == "Exit Game":
            break

    pygame.quit()


def _play_against_ai():
    """Show the opponent picker then launch the appropriate board."""
    choice = ui.pick_ai_opponent()
    if choice is None:
        return  # user backed out
    if choice == "random":
        start.initializing_game(9, True, vs_bot=True)
    else:
        start.initializing_game_vs_nn(weights_path=choice)


def _run_self_play():
    """
    Runs AI self-play in a separate process for instant cancellation.
    """
    import GoGame.pygame_ui as pg_ui

    NUM_GAMES = None
    while NUM_GAMES is None:
        raw = pg_ui.popup_get_text("How many self-play games?", title="AI Self Play", default_text="5")
        if raw is None:
            return
        try:
            NUM_GAMES = max(1, int(raw.strip()))
        except ValueError:
            ui.def_popup("Please enter a valid whole number.", 2)

    manager  = multiprocessing.Manager()
    log_list = manager.list()
    game_counter = manager.list([0])  # [games_completed]

    p = multiprocessing.Process(
        target=_self_play_worker,
        args=(NUM_GAMES, log_list, game_counter),
        daemon=True
    )
    p.start()
    _progress_screen_process(p, log_list, title="AI Self Play",
                             hint=f"Playing {NUM_GAMES} self-play games  •  Esc to cancel",
                             game_counter=game_counter)


def _self_play_worker(num_games, log_list, game_counter):
    """Module-level worker for AI self-play — runs in a separate process."""
    import sys

    class _Log:
        """Redirect all print() output to log_list."""
        def __init__(self, lst): self._lst = lst
        def write(self, s):
            s = s.strip()
            if s and '\r' not in s and '\x1b' not in s:
                self._lst.append(s)
        def flush(self): pass

    sys.stdout = _Log(log_list)
    try:
        from GoGame.neuralnet import training_cycle_process
        training_cycle_process(num_games, log_list, game_counter)
    except Exception as e:
        log_list.append(f"ERROR: {e}")
    finally:
        sys.stdout = sys.__stdout__


def _run_ai_training():
    """
    Runs AI training in a separate process for instant cancellation.
    """
    import os
    if not os.path.exists("saved_self_play.json"):
        import GoGame.pygame_ui as pg_ui
        pg_ui.popup("No training data found.\nRun 'AI Self Play' first.", title="AI Training")
        return

    manager = multiprocessing.Manager()
    log_list = manager.list()
    log_list.append("Starting AI training...")

    p = multiprocessing.Process(
        target=_training_worker,
        args=(log_list,),
        daemon=True
    )
    p.start()
    _progress_screen_process(p, log_list, title="AI Training",
                             hint="Training on self-play data  •  Esc to cancel")


def _run_import_sgf():
    """
    Converts all SGF files in the project-root 'sgf/' folder into
    saved_other_play.json training data using sgf_to_training.py.
    """
    import os

    # resolve sgf/ relative to the project root (three levels up from src/GoGame/main.py)
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sgf_dir    = os.path.join(script_dir, 'sgf')

    if not os.path.isdir(sgf_dir):
        try:
            os.makedirs(sgf_dir)
        except Exception:
            pass
        import GoGame.pygame_ui as pg_ui
        pg_ui.popup(
            f"No 'sgf/' folder found.\nCreated it at:\n{sgf_dir}\n\nAdd your .sgf files there and try again.",
            title="Import SGF Files")
        return

    sgf_files = []
    for root, _, files in os.walk(sgf_dir):
        for fname in files:
            if fname.lower().endswith('.sgf'):
                sgf_files.append(fname)
    if not sgf_files:
        import GoGame.pygame_ui as pg_ui
        pg_ui.popup(f"No .sgf files found in:\n{sgf_dir}", title="Import SGF Files")
        return

    manager      = multiprocessing.Manager()
    log_list     = manager.list()
    game_counter = manager.list([0])
    log_list.append(f"Starting SGF import from {sgf_dir}...")

    p = multiprocessing.Process(
        target=_import_sgf_worker,
        args=(sgf_dir, log_list, game_counter),
        daemon=True
    )
    p.start()
    _progress_screen_process(p, log_list, title="Import SGF Files",
                             hint=f"Importing from sgf/  •  Esc to cancel",
                             game_counter=game_counter)


def _import_sgf_worker(sgf_dir, log_list, game_counter):
    """Module-level worker for SGF import — runs in a separate process."""
    import sys, os
    # ensure GoGame is importable inside the worker process
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from GoGame.sgf_to_training import convert_sgf_dir

    class _Log:
        """Redirect print() inside convert_sgf_dir to log_list."""
        def __init__(self, lst): self._lst = lst
        def write(self, s):
            s = s.strip()
            if s and '\r' not in s and '\x1b' not in s:
                self._lst.append(s)
        def flush(self): pass

    sys.stdout = _Log(log_list)
    try:
        total = convert_sgf_dir(sgf_dir=sgf_dir, verbose=True)
        game_counter[0] = total
    except Exception as e:
        log_list.append(f"ERROR: {e}")
    finally:
        sys.stdout = sys.__stdout__


def _run_sgf_training():
    """
    Trains a model on saved_other_play.json in a separate process for instant cancellation.
    """
    import os
    if not os.path.exists("saved_other_play.json"):
        import GoGame.pygame_ui as pg_ui
        pg_ui.popup("No SGF training data found.\nRun 'Import SGF Files' first.", title="SGF Training")
        return

    manager = multiprocessing.Manager()
    log_list = manager.list()
    log_list.append("Starting SGF model training...")

    p = multiprocessing.Process(
        target=_sgf_training_worker,
        args=(log_list,),
        daemon=True
    )
    p.start()
    _progress_screen_process(p, log_list, title="SGF Training",
                             hint="Training on SGF game data  •  Esc to cancel")


def _training_worker(log_list):
    """Module-level worker for AI training — runs in a separate process."""
    import sys

    class _Log:
        def __init__(self, lst): self._lst = lst
        def write(self, s):
            s = s.strip()
            if s and '\r' not in s and '\x1b' not in s:
                self._lst.append(s)
        def flush(self): pass

    sys.stdout = _Log(log_list)
    try:
        from GoGame.neuralnet import loading_file_for_training
        loading_file_for_training(epochs=10, size_of_batch=32)
    except Exception as e:
        log_list.append(f"ERROR: {e}")
    finally:
        sys.stdout = sys.__stdout__


def _sgf_training_worker(log_list):
    """Module-level worker for SGF training — runs in a separate process."""
    import sys

    class _Log:
        def __init__(self, lst): self._lst = lst
        def write(self, s):
            s = s.strip()
            if s and '\r' not in s and '\x1b' not in s:
                self._lst.append(s)
        def flush(self): pass

    sys.stdout = _Log(log_list)
    try:
        from GoGame.neuralnet import loading_file_for_training_other
        loading_file_for_training_other(epochs=10, size_of_batch=32)
    except Exception as e:
        log_list.append(f"ERROR: {e}")
    finally:
        sys.stdout = sys.__stdout__


def _is_board_row(line: str) -> bool:
    """Return True if every character in the line is a Go stone emoji."""
    if not line:
        return False
    return all(c in ('\u26AB', '\u26AA', '\u26D4') for c in line)


def _draw_pixel_board(screen, board_rows, x, y, cell, last_move_idx=None):
    """
    Draw a list of board row strings as a pixel grid.
    last_move_idx: (row, col) of the last move played, or None.
    """
    _BOARD_BG = (180, 140, 60)
    _BLACK    = (15,  15,  18)
    _WHITE    = (238, 235, 228)
    _LINE     = (100,  75,  30)

    board_size = len(board_rows[0]) if board_rows else 0
    w = board_size * cell
    h = len(board_rows) * cell

    pygame.draw.rect(screen, _BOARD_BG, pygame.Rect(x, y, w, h))

    for i in range(board_size):
        cx = x + i * cell + cell // 2
        cy = y + i * cell + cell // 2
        pygame.draw.line(screen, _LINE, (x + cell // 2, cy), (x + w - cell // 2, cy), 1)
        pygame.draw.line(screen, _LINE, (cx, y + cell // 2), (cx, y + h - cell // 2), 1)

    r = max(2, cell // 2 - 1)
    for row_idx, row in enumerate(board_rows):
        for col_idx, ch in enumerate(row):
            cx = x + col_idx * cell + cell // 2
            cy = y + row_idx * cell + cell // 2
            if ch == '\u26AB':
                pygame.draw.circle(screen, _BLACK, (cx, cy), r)
            elif ch == '\u26AA':
                pygame.draw.circle(screen, _WHITE, (cx, cy), r)

    # last-move indicator — small contrasting dot in the stone centre
    if last_move_idx is not None:
        row_idx, col_idx = last_move_idx
        if 0 <= row_idx < len(board_rows) and 0 <= col_idx < board_size:
            ch = board_rows[row_idx][col_idx]
            cx = x + col_idx * cell + cell // 2
            cy = y + row_idx * cell + cell // 2
            dot_color = (255, 255, 255) if ch == '\u26AB' else (0, 0, 0)
            dot_r = max(1, r // 3)
            pygame.draw.circle(screen, dot_color, (cx, cy), dot_r)


def _progress_screen_process(process, log_list, title: str, hint: str,
                              game_counter=None):
    """
    Progress screen for process-based workers. Terminates process immediately on Escape.
    Board rows are rendered as a pixel grid; other lines are colour-coded text.
    """
    screen = pygame.display.get_surface()
    if screen is None or screen.get_size() != (ui.WIN_W, ui.WIN_H):
        screen = pygame.display.set_mode((ui.WIN_W, ui.WIN_H))
    pygame.display.set_caption(title)

    try:
        f_title = pygame.font.SysFont("arial", 22, bold=True)
        f_hint  = pygame.font.SysFont("arial", 13)
        f_log   = pygame.font.SysFont("arial", 13)
    except Exception:
        f_title = pygame.font.Font(None, 26)
        f_hint  = pygame.font.Font(None, 17)
        f_log   = pygame.font.Font(None, 17)

    _BG      = (35,  30,  25)
    _GOLD    = (220, 185,  90)
    _DIM     = (160, 145, 115)
    _TEXT    = (235, 220, 190)
    _DOT_ON  = (220, 185,  90)
    _DOT_OFF = (80,   70,  50)
    _ERR     = (220,  80,  80)
    _GREEN   = (140, 200, 140)
    _SKIP    = (160, 120,  80)

    LINE_H   = f_log.get_linesize() + 1
    LOG_TOP  = 165
    LOG_BOT  = ui.WIN_H - 10

    def _line_colour(line: str):
        if line.startswith("ERROR"):                           return _ERR
        if line.startswith("[SKIP]"):                         return _SKIP
        if line.startswith("  t=") or ", val=" in line:       return _GREEN
        if "winner" in line.lower() or "wins" in line.lower(): return _GOLD
        if "Weights loaded" in line:                           return _GOLD
        return _DIM

    clock      = pygame.time.Clock()
    anim_frame = 0
    cancelled  = False

    def _render_log(log_lines, y_start, final=False):
        """Render log lines, substituting consecutive board rows with a pixel grid."""
        # extract last move position from the most recent stats line
        last_move = None
        for line in reversed(log_lines):
            if 'pos=' in line and 'val=' in line:
                try:
                    pos_part = line.split('pos=')[1].split()[0]
                    r, c = pos_part.split(',')
                    last_move = (int(r), int(c))
                except Exception:
                    pass
                break

        y = y_start
        i = 0
        while i < len(log_lines):
            if _is_board_row(log_lines[i]):
                # collect all consecutive board rows
                j = i
                while j < len(log_lines) and _is_board_row(log_lines[j]):
                    j += 1
                rows = log_lines[i:j]
                board_size = len(rows[0])
                # cell size: fit within window width with margin, keep square
                max_w = ui.WIN_W - 40
                cell  = max(4, min(max_w // board_size, 28))
                board_h = len(rows) * cell
                if y + board_h > LOG_BOT:
                    break
                bx = (ui.WIN_W - board_size * cell) // 2
                _draw_pixel_board(screen, rows, bx, y, cell, last_move_idx=last_move)
                y += board_h + 4
                i = j
            else:
                if y + LINE_H > LOG_BOT:
                    break
                colour = _ERR if final else _line_colour(log_lines[i])
                ls = f_log.render(log_lines[i], True, colour if not final else _TEXT)
                screen.blit(ls, (20, y))
                y += LINE_H
                i += 1

    while process.is_alive():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                process.kill()
                process.join(timeout=2)
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                process.kill()
                process.join(timeout=2)
                cancelled = True
                break
        else:
            screen.fill(_BG)
            ts = f_title.render(title, True, _GOLD)
            screen.blit(ts, ts.get_rect(centerx=screen.get_width() // 2, y=40))

            anim_frame = (anim_frame + 1) % 60
            dot_cx = screen.get_width() // 2 - 24
            for d in range(4):
                col = _DOT_ON if (anim_frame // 15) == d else _DOT_OFF
                pygame.draw.circle(screen, col, (dot_cx + d * 16, 85), 5)

            hs = f_hint.render(hint, True, _DIM)
            screen.blit(hs, hs.get_rect(centerx=screen.get_width() // 2, y=105))

            if game_counter and game_counter[0] > 0:
                gs = f_hint.render(f"Game {game_counter[0]} complete", True, _TEXT)
                screen.blit(gs, gs.get_rect(centerx=screen.get_width() // 2, y=130))

            # figure out how many recent lines fit, then render
            all_lines = list(log_list)
            # show as many recent lines as fit — board grids take multiple rows so
            # we can't know exactly without rendering; use last 60 lines as a window
            _render_log(all_lines[-20:], LOG_TOP)

            pygame.display.flip()
            clock.tick(30)
            continue
        break

    # done — show final result and wait for keypress
    final_log = list(log_list)
    screen.fill(_BG)
    ts = f_title.render("Cancelled" if cancelled else "Done!", True, _GOLD)
    screen.blit(ts, ts.get_rect(centerx=screen.get_width() // 2, y=40))
    _render_log(final_log[-20:], 100, final=True)
    hs = f_hint.render("Press any key to return to menu", True, _DIM)
    screen.blit(hs, hs.get_rect(centerx=screen.get_width() // 2,
                                 y=screen.get_height() - 60))
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.QUIT):
                waiting = False
        clock.tick(30)


if __name__ == "__main__":
    pygame.init()
    play_game_main()
