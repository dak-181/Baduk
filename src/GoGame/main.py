import pygame
import threading
import GoGame.game_initialization as start
import GoGame.uifunctions as ui


def play_game_main():
    '''
    Main menu loop. Uses ui.start_game_menu() which returns the chosen
    action as a string.
    '''
    while True:
        event = ui.start_game_menu()

        if event == "Choose File":
            from GoGame.saving_loading import choose_file
            choose_file()
            break

        elif event == "New Game From Custom":
            board_size = ui.start_game()
            start.initializing_game(board_size, defaults=False)

        elif event == "New Game From Default":
            start.initializing_game(9, True)

        elif event == "Play Against AI":
            start.initializing_game(9, True, vs_bot=True)

        elif event == "AI SelfPlay":
            _run_self_play()

        elif event == "AI Training":
            _run_ai_training()

        elif event == "Exit Game":
            break

    pygame.quit()


def _run_self_play():
    """
    Runs AI self-play in a background thread while showing a
    live progress screen in pygame. Press Escape to cancel early.
    """
    from GoGame.neuralnet import training_cycle

    NUM_GAMES = 5
    state = {"done": False, "cancelled": False, "game": 0, "log": []}

    def _worker():
        try:
            training_cycle(NUM_GAMES, state)
        except Exception as e:
            state["log"].append(f"ERROR: {e}")
        finally:
            state["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    _progress_screen(t, state, title="AI Self Play",
                     hint=f"Playing {NUM_GAMES} self-play games  •  Esc to cancel")
    state["cancelled"] = True


def _run_ai_training():
    """
    Runs AI training in a background thread with a progress screen.
    """
    import os
    if not os.path.exists("saved_self_play.json"):
        ui.def_popup("No training data found.\nRun 'AI Self Play' first.", 4)
        return

    from GoGame.neuralnet import loading_file_for_training

    state = {"done": False, "cancelled": False, "log": []}

    def _worker():
        try:
            loading_file_for_training(epochs=10, size_of_batch=32, state=state)
        except Exception as e:
            state["log"].append(f"ERROR: {e}")
        finally:
            state["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    _progress_screen(t, state, title="AI Training",
                     hint="Training on self-play data  •  Esc to cancel")
    state["cancelled"] = True


def _progress_screen(thread, state: dict, title: str, hint: str):
    """
    Draws a live progress/log screen while the worker thread runs.
    Returns when done or user presses Escape.
    """
    screen = pygame.display.get_surface()
    if screen is None:
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

    _BG      = (35, 30, 25)
    _GOLD    = (220, 185, 90)
    _DIM     = (160, 145, 115)
    _TEXT    = (235, 220, 190)
    _DOT_ON  = (220, 185, 90)
    _DOT_OFF = (80, 70, 50)

    clock      = pygame.time.Clock()
    anim_frame = 0
    MAX_LOG    = 18

    while not state["done"]:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                state["cancelled"] = True
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                state["cancelled"] = True
                return

        screen.fill(_BG)

        ts = f_title.render(title, True, _GOLD)
        screen.blit(ts, ts.get_rect(centerx=screen.get_width() // 2, y=40))

        # animated spinner dots
        anim_frame = (anim_frame + 1) % 60
        dot_cx = screen.get_width() // 2 - 24
        for d in range(4):
            col = _DOT_ON if (anim_frame // 15) == d else _DOT_OFF
            pygame.draw.circle(screen, col, (dot_cx + d * 16, 85), 5)

        hs = f_hint.render(hint, True, _DIM)
        screen.blit(hs, hs.get_rect(centerx=screen.get_width() // 2, y=105))

        if "game" in state and state["game"] > 0:
            gs = f_hint.render(f"Game {state['game']} complete", True, _TEXT)
            screen.blit(gs, gs.get_rect(centerx=screen.get_width() // 2, y=130))

        log_lines = state.get("log", [])[-MAX_LOG:]
        log_y = 165
        for line in log_lines:
            ls = f_log.render(line[:100], True, _DIM)
            screen.blit(ls, (40, log_y))
            log_y += f_log.get_linesize() + 2

        pygame.display.flip()
        clock.tick(30)

    # done — show final log and wait for keypress
    screen.fill(_BG)
    ts = f_title.render("Done!" if not state.get("cancelled") else "Cancelled", True, _GOLD)
    screen.blit(ts, ts.get_rect(centerx=screen.get_width() // 2, y=40))
    log_y = 100
    for line in state.get("log", [])[-MAX_LOG:]:
        ls = f_log.render(line[:100], True, _TEXT)
        screen.blit(ls, (40, log_y))
        log_y += f_log.get_linesize() + 2
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
