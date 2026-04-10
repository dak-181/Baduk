"""
uifunctions.py  (modernised — no PySimpleGUI)
----------------------------------------------
All GUI is now pure pygame.  The separate PySimpleGUI window that used to
embed pygame inside a sg.Graph is gone; instead we run a single full-screen
pygame window that renders the board *and* the sidebar.

Layout
──────
  ┌──────────────────────────────────────────┐
  │  [Pass] [Save] [Undo] [Quit] [Exit Menu] │  ← button bar  (60 px)
  ├───────────────────┬──────────────────────┤
  │                   │                      │
  │   board area      │    sidebar           │
  │   (700 × 700)     │    (270 × 700)       │
  │                   │                      │
  └───────────────────┴──────────────────────┘
  Total window: 970 × 760
"""

import os
import platform
import pygame
from typing import Tuple, List, Optional
import GoGame.config as cf
import GoGame.pygame_ui as ui
import pygame.gfxdraw

# ── window dimensions ────────────────────────────────────────────────────────
BOARD_W, BOARD_H = 700, 700
SIDE_W            = 270
BTN_BAR_H         = 60
WIN_W             = BOARD_W + SIDE_W
WIN_H             = BOARD_H + BTN_BAR_H

# ── colours ──────────────────────────────────────────────────────────────────
_BOARD_BG   = (220, 179, 92)
_SIDE_BG    = (40,   36,  30)
_BAR_BG     = (50,   45,  38)
_BAR_BORDER = (180, 140,  60)
_TEXT_COL   = (235, 220, 190)
_TEXT_DIM   = (160, 145, 115)
_BTN_NRM    = (80,   68,  42)
_BTN_HOV    = (120, 100,  52)
_BTN_TXT    = (240, 225, 180)
_SEPARATOR  = (100,  88,  60)

# ── button definitions ───────────────────────────────────────────────────────
_BUTTONS = ["Pass Turn", "Save Game", "Undo Turn", "Quit Program", "Exit To Menu"]


def _font(size: int = 16, bold: bool = False) -> pygame.font.Font:
    try:
        return pygame.font.SysFont("arial", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size + 4)


# ── pygame setup ─────────────────────────────────────────────────────────────

def _ensure_pygame() -> None:
    if not pygame.get_init():
        pygame.init()


def _make_window() -> pygame.Surface:
    _ensure_pygame()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("GoGame")
    return screen


# ── button-bar helpers ───────────────────────────────────────────────────────

def _button_rects() -> dict:
    """Return a label → pygame.Rect mapping for the button bar."""
    n      = len(_BUTTONS)
    margin = 8
    total  = WIN_W - margin * (n + 1)
    bw     = total // n
    bh     = BTN_BAR_H - 14
    rects  = {}
    for i, label in enumerate(_BUTTONS):
        x = margin + i * (bw + margin)
        y = 7
        rects[label] = pygame.Rect(x, y, bw, bh)
    return rects


def _draw_button_bar(screen: pygame.Surface,
                     btn_rects: dict,
                     mode: str = "Playing") -> None:
    bar = pygame.Rect(0, 0, WIN_W, BTN_BAR_H)
    pygame.draw.rect(screen, _BAR_BG, bar)
    pygame.draw.line(screen, _BAR_BORDER, (0, BTN_BAR_H - 1), (WIN_W, BTN_BAR_H - 1), 1)
    mx, my  = pygame.mouse.get_pos()
    f       = _font(13, bold=True)
    for label, rect in btn_rects.items():
        disp  = label
        # in scoring mode "Quit Program" becomes "Resume Game"
        if label == "Quit Program" and mode == "Scoring":
            disp = "Resume Game"
        hov   = rect.collidepoint(mx, my)
        colour = _BTN_HOV if hov else _BTN_NRM
        pygame.draw.rect(screen, colour, rect, border_radius=5)
        pygame.draw.rect(screen, _BAR_BORDER, rect, width=1, border_radius=5)
        ts = f.render(disp, True, _BTN_TXT)
        screen.blit(ts, ts.get_rect(center=rect.center))


def _draw_sidebar(screen: pygame.Surface, game_board) -> None:
    sx = BOARD_W
    pygame.draw.rect(screen, _SIDE_BG, pygame.Rect(sx, BTN_BAR_H, SIDE_W, BOARD_H))
    pygame.draw.line(screen, _SEPARATOR, (sx, BTN_BAR_H), (sx, WIN_H), 1)

    pb = game_board.player_black
    pw = game_board.player_white
    lines = [
        ("Turn",    f"{game_board.whose_turn.color}",     True),
        ("",        f"Turn #{game_board.turn_num}",        False),
        ("",        "",                                    False),
        ("Black",   pb.name,                               True),
        ("",        f"komi: {pb.komi}",                    False),
        ("",        "",                                    False),
        ("White",   pw.name,                               True),
        ("",        f"komi: {pw.komi}",                    False),
    ]

    f_lbl  = _font(13, bold=True)
    f_val  = _font(13)
    y      = BTN_BAR_H + 16
    x      = sx + 12

    for lbl, val, bold in lines:
        if lbl:
            ts = f_lbl.render(f"{lbl}: ", True, _BAR_BORDER)
            screen.blit(ts, (x, y))
            ts2 = (f_lbl if bold else f_val).render(val, True, _TEXT_COL)
            screen.blit(ts2, (x + ts.get_width(), y))
        else:
            ts = f_val.render(val, True, _TEXT_DIM)
            screen.blit(ts, (x, y))
        y += f_val.get_linesize() + 3


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN-MENU screen (replaces setup_menu + its event loop)
# ═══════════════════════════════════════════════════════════════════════════════

def start_game_menu() -> str:
    """
    Draws the main menu and returns the chosen action string:
      'Choose File' | 'New Game From Custom' | 'New Game From Default'
    | 'Play Against AI' | 'AI SelfPlay' | 'AI Training' | 'Exit Game'
    """
    screen = _make_window()
    clock  = pygame.time.Clock()

    menu_buttons = [
        "Choose File",
        "New Game From Custom",
        "New Game From Default",
        "Play Against AI",
        "AI SelfPlay",
        "AI Training",
        "Exit Game",
    ]

    f_title = _font(26, bold=True)
    f_sub   = _font(14)
    f_btn   = _font(16, bold=True)

    btn_w, btn_h = 280, 46
    gap          = 12
    start_y      = 220

    def _btn_rects():
        rects = {}
        for i, lbl in enumerate(menu_buttons):
            x = WIN_W // 2 - btn_w // 2
            y = start_y + i * (btn_h + gap)
            rects[lbl] = pygame.Rect(x, y, btn_w, btn_h)
        return rects

    while True:
        mx, my = pygame.mouse.get_pos()
        screen.fill((35, 30, 25))

        ts_t = f_title.render("Go Go Go", True, (220, 185, 90))
        screen.blit(ts_t, ts_t.get_rect(centerx=WIN_W // 2, y=80))
        ts_s = f_sub.render(
            "9×9 default · 7.5 komi · Player 1 vs Player 2",
            True, _TEXT_DIM)
        screen.blit(ts_s, ts_s.get_rect(centerx=WIN_W // 2, y=140))

        rects = _btn_rects()
        for lbl, rect in rects.items():
            hov = rect.collidepoint(mx, my)
            pygame.draw.rect(screen, _BTN_HOV if hov else _BTN_NRM, rect, border_radius=8)
            pygame.draw.rect(screen, _BAR_BORDER, rect, width=1, border_radius=8)
            ts = f_btn.render(lbl, True, _BTN_TXT)
            screen.blit(ts, ts.get_rect(center=rect.center))

        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "Exit Game"
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for lbl, rect in rects.items():
                    if rect.collidepoint(ev.pos):
                        return lbl
        clock.tick(60)


# ═══════════════════════════════════════════════════════════════════════════════
#  BOARD-SIZE chooser  (was start_game)
# ═══════════════════════════════════════════════════════════════════════════════

def start_game() -> int:
    """Ask the player to choose board size. Returns 9, 13, or 19."""
    screen = pygame.display.get_surface() or _make_window()
    snapshot = screen.copy()
    clock = pygame.time.Clock()

    options = [("9×9", 9), ("13×13", 13), ("19×19", 19)]
    f       = _font(20, bold=True)
    f_info  = _font(15)
    btn_w, btn_h = 120, 52
    gap  = 24
    cx   = screen.get_width() // 2
    cy   = screen.get_height() // 2

    total  = btn_w * 3 + gap * 2
    starts = [cx - total // 2 + i * (btn_w + gap) for i in range(3)]
    rects  = [(pygame.Rect(x, cy + 20, btn_w, btn_h), lbl, val)
              for (lbl, val), x in zip(options, starts)]

    panel_r = pygame.Rect(cx - 260, cy - 60, 520, 170)

    while True:
        mx, my = pygame.mouse.get_pos()
        screen.blit(snapshot, (0, 0))
        _overlay_dark(screen)
        pygame.draw.rect(screen, (55, 48, 38), panel_r, border_radius=12)
        pygame.draw.rect(screen, _BAR_BORDER,  panel_r, width=2, border_radius=12)
        ts = f_info.render("Choose board size:", True, _TEXT_COL)
        screen.blit(ts, ts.get_rect(centerx=cx, y=cy - 36))
        for rect, lbl, _ in rects:
            hov = rect.collidepoint(mx, my)
            pygame.draw.rect(screen, _BTN_HOV if hov else _BTN_NRM, rect, border_radius=8)
            pygame.draw.rect(screen, _BAR_BORDER, rect, width=1, border_radius=8)
            ts = f.render(lbl, True, _BTN_TXT)
            screen.blit(ts, ts.get_rect(center=rect.center))
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); import sys; sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for rect, _, val in rects:
                    if rect.collidepoint(ev.pos):
                        screen.blit(snapshot, (0, 0))
                        pygame.display.flip()
                        return val
        clock.tick(60)


def _overlay_dark(surf: pygame.Surface) -> None:
    ov = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 150))
    surf.blit(ov, (0, 0))


# ═══════════════════════════════════════════════════════════════════════════════
#  HANDICAP choosers
# ═══════════════════════════════════════════════════════════════════════════════

def handicap_person_gui() -> str:
    """Returns 'Black', 'White', or \"I don't want a handicap\"."""
    return _button_choice_dialog(
        "Which player gets a handicap?",
        ["Black", "White", "I don't want a handicap"]
    )


def handicap_number_gui(board_size: int) -> int:
    """Returns the chosen handicap integer (1-9, capped at 5 for 9x9)."""
    max_h = 5 if board_size == 9 else 9
    labels = [str(i) for i in range(1, max_h + 1)]
    result = _button_choice_dialog("How large is the handicap?", labels)
    return int(result)


def _button_choice_dialog(prompt: str, options: list) -> str:
    """Generic button-grid popup. Returns the label of the clicked button."""
    screen   = pygame.display.get_surface() or _make_window()
    snapshot = screen.copy()
    clock    = pygame.time.Clock()
    f_info   = _font(15)
    f_btn    = _font(15, bold=True)
    cx, cy   = screen.get_width() // 2, screen.get_height() // 2

    # lay out in rows of up to 4
    per_row  = 4
    btn_w, btn_h = 160, 46
    gap = 12
    rows = [options[i:i+per_row] for i in range(0, len(options), per_row)]
    total_h = len(rows) * (btn_h + gap) - gap + 60
    panel_r = pygame.Rect(cx - 340, cy - total_h // 2 - 20, 680, total_h + 50)

    def _rects():
        rs = []
        for ri, row in enumerate(rows):
            row_total = btn_w * len(row) + gap * (len(row) - 1)
            ox = cx - row_total // 2
            for ci, lbl in enumerate(row):
                x = ox + ci * (btn_w + gap)
                y = panel_r.y + 50 + ri * (btn_h + gap)
                rs.append((pygame.Rect(x, y, btn_w, btn_h), lbl))
        return rs

    result = [None]
    while result[0] is None:
        mx, my = pygame.mouse.get_pos()
        screen.blit(snapshot, (0, 0))
        _overlay_dark(screen)
        pygame.draw.rect(screen, (55, 48, 38), panel_r, border_radius=12)
        pygame.draw.rect(screen, _BAR_BORDER,  panel_r, width=2, border_radius=12)
        ts = f_info.render(prompt, True, _TEXT_COL)
        screen.blit(ts, ts.get_rect(centerx=cx, y=panel_r.y + 14))
        for rect, lbl in _rects():
            hov = rect.collidepoint(mx, my)
            pygame.draw.rect(screen, _BTN_HOV if hov else _BTN_NRM, rect, border_radius=7)
            pygame.draw.rect(screen, _BAR_BORDER, rect, width=1, border_radius=7)
            ts = f_btn.render(lbl, True, _BTN_TXT)
            screen.blit(ts, ts.get_rect(center=rect.center))
        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); import sys; sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for rect, lbl in _rects():
                    if rect.collidepoint(ev.pos):
                        result[0] = lbl
        clock.tick(60)

    screen.blit(snapshot, (0, 0))
    pygame.display.flip()
    return result[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  BOARD WINDOW setup  (replaces setup_board_window_pygame)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_board_window_pygame(game_board) -> None:
    """
    Initialise (or re-use) the pygame window for the game board.
    Stores screen + button_rects on game_board so the game loop can use them.
    (No return value — replaces the old sg.Window return.)
    """
    from GoGame.stone_renderer import clear_cache
    clear_cache()

    _ensure_pygame()
    screen = pygame.display.get_surface()
    if screen is None or screen.get_size() != (WIN_W, WIN_H):
        screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("GoGame")

    game_board.screen      = screen
    game_board.window      = None          # sentinel: no sg.Window anymore
    game_board.btn_rects   = _button_rects()

    screen.fill((30, 27, 22))
    _draw_button_bar(screen, game_board.btn_rects, game_board.mode)
    _draw_sidebar(screen, game_board)

    board_surf = pygame.Surface((BOARD_W, BOARD_H))
    board_surf.fill(_BOARD_BG)
    draw_gameboard(game_board, board_surf)
    screen.blit(board_surf, (0, BTN_BAR_H))
    game_board.backup_board = board_surf
    pygame.display.flip()


# ═══════════════════════════════════════════════════════════════════════════════
#  POPUPS / info helpers  (thin wrappers around pygame_ui)
# ═══════════════════════════════════════════════════════════════════════════════

def validation_gui(info1: str, var_type):
    """Prompt with a text-input dialog until a valid value is entered."""
    output = None
    while output is None:
        output = ui.popup_get_text(info1, title="Enter Information")
    return var_type(output)


def update_scoring(board) -> None:
    """Redraw the sidebar with fresh info (called every turn)."""
    screen = pygame.display.get_surface()
    if screen is None:
        return
    _draw_sidebar(screen, board)
    pygame.display.flip()


def scoring_mode_popup() -> None:
    ui.popup(
        "Click stones you believe are dead, then pass twice to finish scoring.",
        title="Scoring Mode",
        auto_close=True, auto_close_duration=3
    )


def end_game_popup(board) -> None:
    pb = board.player_black
    pw = board.player_white
    bs = pb.komi + pb.territory + pb.black_set_len
    ws = pw.komi + pw.territory + pw.white_set_len
    diff = bs - ws
    winner = (f"Black wins by {diff} pts" if diff > 0 else f"White wins by {-diff} pts")
    msg = (f"Game over!\n\n"
           f"Black ({pb.name}):  territory {pb.territory} + "
           f"captures {pb.black_set_len} + komi {pb.komi} = {bs}\n"
           f"White ({pw.name}):  territory {pw.territory} + "
           f"captures {pw.white_set_len} + komi {pw.komi} = {ws}\n\n"
           f"{winner}")
    ui.popup(msg, title="Game Concluded", auto_close=True, auto_close_duration=20)


def default_popup_no_button(info: str, time: float) -> None:
    ui.popup_no_buttons(info, auto_close_duration=time, non_blocking=True)


def def_popup(info: str, time: float) -> None:
    ui.popup(info, auto_close=True, auto_close_duration=time)


# ═══════════════════════════════════════════════════════════════════════════════
#  BOARD DRAWING  (unchanged logic, adapted surface target)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_gameboard(game_board, surface: Optional[pygame.Surface] = None) -> None:
    """Draw the empty board grid + stars onto *surface* (defaults to backup_board)."""
    if surface is None:
        surface = getattr(game_board, "backup_board", None)
    if surface is None:
        return
    workable_area = 620
    distance      = workable_area / (game_board.board_size - 1)
    circle_radius = distance / 2.1
    game_board.pygame_board_vals = (workable_area, distance, circle_radius)
    surface.fill(_BOARD_BG)
    draw_lines(game_board, distance, surface)
    stars_pygame(game_board, surface, max(2, int(distance * 0.06)))
    draw_coordinates(game_board, surface)


def draw_lines(game_board, distance: float, surface: pygame.Surface) -> None:
    for xidx in range(game_board.board_size):
        x_val = 40 + xidx * distance
        for yidx in range(game_board.board_size):
            y_val = 40 + yidx * distance
            if xidx > 0:
                pygame.draw.line(surface, (0, 0, 0),
                                 (x_val - distance, y_val), (x_val, y_val))
            if yidx > 0:
                pygame.draw.line(surface, (0, 0, 0),
                                 (x_val, y_val - distance), (x_val, y_val))


def stars_pygame(board, surface, circle_radius):
    size = board.board_size
    if size == 9:
        pts = [(2, 2), (size-3, 2), (size-3, size-3), (2, size-3)]
    else:
        pts = [(3, 3), (size-4, 3), (size-4, size-4), (3, size-4)]
    for r, c in pts:
        node = board.board[r][c]
        x = int(node.screen_row)
        y = int(node.screen_col)
        r_int = int(circle_radius)
        pygame.gfxdraw.aacircle(surface, x, y, r_int, (0, 0, 0))
        pygame.gfxdraw.filled_circle(surface, x, y, r_int, (0, 0, 0))
    
        

def draw_coordinates(game_board, surface: pygame.Surface) -> None:
    """Draw A-T column labels and 1-19 row numbers around the board."""
    f = _font(11)
    size = game_board.board_size
    distance = game_board.pygame_board_vals[1]
    cols = "ABCDEFGHJKLMNOPQRST"  # no 'I', standard Go notation

    for i in range(size):
        val = 40 + i * distance

        # column letters — top and bottom
        lbl = f.render(cols[i], True, (0, 0, 0))
        surface.blit(lbl, lbl.get_rect(centerx=val, centery=18))
        surface.blit(lbl, lbl.get_rect(centerx=val, centery=682))

        # row numbers — left and right (1 at bottom, n at top)
        num = f.render(str(size - i), True, (0, 0, 0))
        surface.blit(num, num.get_rect(centerx=18, centery=val))
        surface.blit(num, num.get_rect(centerx=682, centery=val))


# ═══════════════════════════════════════════════════════════════════════════════
#  BOARD REFRESH  (blits backup_board + stones to the screen)
# ═══════════════════════════════════════════════════════════════════════════════

def refresh_board_pygame(board) -> None:
    """Redraw stones on top of the background board surface, then flip."""
    from GoGame.stone_renderer import draw_stone

    screen = board.screen
    if screen is None:
        return

    screen.blit(board.backup_board, (0, BTN_BAR_H))
    radius = int(board.pygame_board_vals[1] / 2.1) * 2

    for row in board.board:
        for node in row:
            c  = node.stone_here_color
            sx = int(node.screen_row)
            sy = int(node.screen_col) + BTN_BAR_H

            if c == cf.rgb_black:
                draw_stone(screen, 'black', sx, sy, radius)
            elif c == cf.rgb_white:
                draw_stone(screen, 'white', sx, sy, radius)
            elif c == cf.rgb_peach:
                draw_stone(screen, 'black', sx, sy, radius, alpha=128)
            elif c == cf.rgb_lavender:
                draw_stone(screen, 'white', sx, sy, radius, alpha=128)
            elif c == cf.rgb_green:
                draw_stone(screen, 'territory_black', sx, sy, radius)
            elif c == cf.rgb_red:
                draw_stone(screen, 'territory_white', sx, sy, radius)

    # last-move marker
    last = getattr(board, 'last_move', None)
    if last is not None and last.stone_here_color in (cf.rgb_black, cf.rgb_white):
        marker_color = (255, 255, 255) if last.stone_here_color == cf.rgb_black else (0, 0, 0)
        stone_col    = (15, 15, 18) if last.stone_here_color == cf.rgb_black else (238, 235, 228)
        ring_r  = max(2, int(board.pygame_board_vals[1] * 0.18))
        inner_r = max(1, int(ring_r * 0.75))
        mx = int(last.screen_row)
        my = int(last.screen_col) + BTN_BAR_H
        pygame.gfxdraw.aacircle(screen, mx, my, ring_r, marker_color)
        pygame.gfxdraw.filled_circle(screen, mx, my, ring_r, marker_color)
        pygame.gfxdraw.aacircle(screen, mx, my, inner_r, stone_col)
        pygame.gfxdraw.filled_circle(screen, mx, my, inner_r, stone_col)

    _draw_button_bar(screen, board.btn_rects, board.mode)
    _draw_sidebar(screen, board)
    pygame.display.flip()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE SWITCH  (updates button label; replaces switch_button_mode)
# ═══════════════════════════════════════════════════════════════════════════════

def switch_button_mode(board) -> None:
    if board.mode == "Scoring":
        board.times_passed = 0
    board.mode_change = False
    refresh_board_pygame(board)


# ═══════════════════════════════════════════════════════════════════════════════
#  WINDOW CLOSE
# ═══════════════════════════════════════════════════════════════════════════════

def close_window(board) -> None:
    """Release pygame display resources (called before returning to menu)."""
    try:
        del board.backup_board
    except AttributeError:
        pass
    board.screen = None
    board.window = None
    pygame.display.quit()
    pygame.display.init()   # keep pygame alive for the menu
