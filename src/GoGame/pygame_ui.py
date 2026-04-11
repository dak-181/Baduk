"""
pygame_ui.py
------------
Drop-in replacement for every PySimpleGUI call in GoGame.
All dialogs are drawn directly onto the pygame display — no external
dependencies, works perfectly with PyInstaller --onefile.

Public API (mirrors the sg.* calls used in the codebase):
    popup(msg, *, title="", auto_close_duration=3)
    popup_no_buttons(msg, *, auto_close_duration=3)
    popup_yes_no(msg, *, title="") -> "Yes" | "No"
    popup_get_text(msg, *, title="") -> str | None
    popup_get_file(msg, *, title="") -> str | None
    WIN_CLOSED = None   (sentinel, same as sg.WIN_CLOSED)
"""

import pygame
import os
import sys
from typing import Optional

# ── sentinel matches sg.WIN_CLOSED ──────────────────────────────────────────
WIN_CLOSED = None

# ── colours (flat, dark-amber-ish to feel familiar) ─────────────────────────
_BG        = (45,  40,  35)
_PANEL     = (62,  55,  48)
_BORDER    = (180, 140,  60)
_TEXT      = (235, 220, 190)
_TEXT_DIM  = (160, 145, 115)
_BTN_NRM   = (90,  75,  45)
_BTN_HOV   = (130, 105,  55)
_BTN_TXT   = (240, 225, 180)
_INPUT_BG  = (30,  27,  22)
_INPUT_ACT = (50,  45,  35)
_CURSOR    = (220, 200, 140)

# ── typography ───────────────────────────────────────────────────────────────
_FONT_CACHE: dict = {}

def _font(size: int = 18, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = pygame.font.SysFont("arial", size, bold=bold)
        except Exception:
            _FONT_CACHE[key] = pygame.font.Font(None, size + 4)
    return _FONT_CACHE[key]


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_display() -> pygame.Surface:
    """Return the current display surface, initialising pygame if needed."""
    if not pygame.get_init():
        pygame.init()
    surf = pygame.display.get_surface()
    if surf is None:
        surf = pygame.display.set_mode((800, 700))
        pygame.display.set_caption("GoGame")
    return surf


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Word-wrap *text* to fit within *max_width* pixels."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            test = current + " " + word
            if font.size(test)[0] <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _draw_panel(surf: pygame.Surface, rect: pygame.Rect, title: str = "") -> None:
    """Draw a modal panel (rounded rect + optional title bar)."""
    pygame.draw.rect(surf, _PANEL, rect, border_radius=10)
    pygame.draw.rect(surf, _BORDER, rect, width=2, border_radius=10)
    if title:
        tf = _font(15, bold=True)
        ts = tf.render(title, True, _BORDER)
        surf.blit(ts, (rect.x + 12, rect.y + 10))


def _draw_button(surf: pygame.Surface, rect: pygame.Rect,
                 label: str, hovered: bool = False) -> None:
    colour = _BTN_HOV if hovered else _BTN_NRM
    pygame.draw.rect(surf, colour, rect, border_radius=6)
    pygame.draw.rect(surf, _BORDER, rect, width=1, border_radius=6)
    f = _font(16, bold=True)
    ts = f.render(label, True, _BTN_TXT)
    surf.blit(ts, ts.get_rect(center=rect.center))


def _overlay(surf: pygame.Surface) -> pygame.Surface:
    """Return a semi-transparent dark overlay blit'd onto *surf*."""
    ov = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 160))
    surf.blit(ov, (0, 0))
    return surf


def _pump_and_wait_close(timeout_ms: Optional[int] = None) -> None:
    """Drain the event queue; block until the popup closes itself (timeout)."""
    import time
    start = time.time()
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.flip()
        clock.tick(30)
        if timeout_ms and (time.time() - start) * 1000 >= timeout_ms:
            break


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def popup(msg: str, *, title: str = "", auto_close: bool = False,
          auto_close_duration: float = 3, line_width: int = 42,
          non_blocking: bool = False, font=None) -> None:
    """Show an info popup with an OK button (or auto-close)."""
    import threading
    import time

    surf = _get_display()
    snapshot = surf.copy()

    panel_w, panel_h = 480, 0   # height computed after wrapping
    f_body  = _font(16)
    f_title = _font(15, bold=True)
    pad = 20
    btn_h   = 40
    btn_w   = 100

    wrapped = _wrap_text(msg, f_body, panel_w - pad * 2)
    text_h  = len(wrapped) * (f_body.get_linesize() + 2)
    title_h = (f_title.get_linesize() + 12) if title else 0
    panel_h = pad + title_h + text_h + pad + btn_h + pad

    cx, cy  = surf.get_width() // 2, surf.get_height() // 2
    panel_r = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)
    btn_r   = pygame.Rect(cx - btn_w // 2, panel_r.bottom - pad - btn_h, btn_w, btn_h)

    closed  = [False]

    def _draw(hov: bool) -> None:
        surf.blit(snapshot, (0, 0))
        _overlay(surf)
        _draw_panel(surf, panel_r, title)
        ty = panel_r.y + pad + title_h
        for line in wrapped:
            ts = f_body.render(line, True, _TEXT)
            surf.blit(ts, (panel_r.x + pad, ty))
            ty += f_body.get_linesize() + 2
        _draw_button(surf, btn_r, "OK", hov)
        pygame.display.flip()

    if non_blocking:
        def _auto():
            time.sleep(auto_close_duration)
            surf.blit(snapshot, (0, 0))
            pygame.display.flip()
        threading.Thread(target=_auto, daemon=True).start()
        _draw(False)
        return

    deadline = time.time() + auto_close_duration if auto_close else None
    clock    = pygame.time.Clock()
    while not closed[0]:
        mx, my = pygame.mouse.get_pos()
        hov    = btn_r.collidepoint(mx, my)
        _draw(hov)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and hov:
                closed[0] = True
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
                closed[0] = True
        if deadline and time.time() >= deadline:
            closed[0] = True
        clock.tick(30)

    surf.blit(snapshot, (0, 0))
    pygame.display.flip()


def popup_no_buttons(msg: str, *, title: str = "", auto_close: bool = True,
                     auto_close_duration: float = 3, non_blocking: bool = True,
                     font=None) -> None:
    """Auto-closing info message with no buttons. Blocks for auto_close_duration seconds."""
    import time

    surf     = _get_display()
    snapshot = surf.copy()

    panel_w  = 480
    f_body   = _font(16)
    f_title  = _font(15, bold=True)
    pad      = 20

    wrapped  = _wrap_text(msg, f_body, panel_w - pad * 2)
    text_h   = len(wrapped) * (f_body.get_linesize() + 2)
    title_h  = (f_title.get_linesize() + 12) if title else 0
    panel_h  = pad + title_h + text_h + pad

    cx, cy   = surf.get_width() // 2, surf.get_height() // 2
    panel_r  = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)

    def _draw() -> None:
        surf.blit(snapshot, (0, 0))
        _overlay(surf)
        _draw_panel(surf, panel_r, title)
        ty = panel_r.y + pad + title_h
        for line in wrapped:
            ts = f_body.render(line, True, _TEXT)
            surf.blit(ts, (panel_r.x + pad, ty))
            ty += f_body.get_linesize() + 2
        pygame.display.flip()

    deadline = time.time() + auto_close_duration
    clock    = pygame.time.Clock()
    while time.time() < deadline:
        _draw()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        clock.tick(30)

    surf.blit(snapshot, (0, 0))
    pygame.display.flip()


def popup_yes_no(msg: str, *, title: str = "", font=None) -> str:
    """Modal yes/no dialog. Returns 'Yes' or 'No'."""
    surf    = _get_display()
    snapshot = surf.copy()

    panel_w = 480
    f_body  = _font(16)
    f_title = _font(15, bold=True)
    pad     = 20
    btn_h, btn_w = 42, 110

    wrapped = _wrap_text(msg, f_body, panel_w - pad * 2)
    text_h  = len(wrapped) * (f_body.get_linesize() + 2)
    title_h = (f_title.get_linesize() + 12) if title else 0
    panel_h = pad + title_h + text_h + pad + btn_h + pad

    cx, cy  = surf.get_width() // 2, surf.get_height() // 2
    panel_r = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)

    gap     = 20
    total   = btn_w * 2 + gap
    yes_r   = pygame.Rect(cx - total // 2,           panel_r.bottom - pad - btn_h, btn_w, btn_h)
    no_r    = pygame.Rect(cx - total // 2 + btn_w + gap, panel_r.bottom - pad - btn_h, btn_w, btn_h)

    result  = [None]
    clock   = pygame.time.Clock()

    while result[0] is None:
        mx, my = pygame.mouse.get_pos()
        surf.blit(snapshot, (0, 0))
        _overlay(surf)
        _draw_panel(surf, panel_r, title)
        ty = panel_r.y + pad + title_h
        for line in wrapped:
            ts = f_body.render(line, True, _TEXT)
            surf.blit(ts, (panel_r.x + pad, ty))
            ty += f_body.get_linesize() + 2
        _draw_button(surf, yes_r, "Yes", yes_r.collidepoint(mx, my))
        _draw_button(surf, no_r,  "No",  no_r.collidepoint(mx, my))
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if yes_r.collidepoint(ev.pos): result[0] = "Yes"
                if no_r.collidepoint(ev.pos):  result[0] = "No"
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_y: result[0] = "Yes"
                if ev.key == pygame.K_n: result[0] = "No"
                if ev.key == pygame.K_ESCAPE: result[0] = "No"
        clock.tick(30)

    surf.blit(snapshot, (0, 0))
    pygame.display.flip()
    return result[0]


def popup_get_text(msg: str, *, title: str = "", font=None,
                   default_text: str = "",
                   ok_label: str = "OK",
                   cancel_label: str = "Cancel") -> Optional[str]:
    """
    Single-line text-input dialog.
    Returns the entered string, or None if the user dismissed/cancelled.
    """
    surf     = _get_display()
    snapshot = surf.copy()

    panel_w  = 500
    f_body   = _font(16)
    f_input  = _font(17)
    f_title  = _font(15, bold=True)
    pad      = 20
    inp_h    = 38
    btn_h, btn_w = 42, 110

    wrapped  = _wrap_text(msg, f_body, panel_w - pad * 2)
    text_h   = len(wrapped) * (f_body.get_linesize() + 2)
    title_h  = (f_title.get_linesize() + 12) if title else 0
    panel_h  = pad + title_h + text_h + 12 + inp_h + pad + btn_h + pad

    cx, cy   = surf.get_width() // 2, surf.get_height() // 2
    panel_r  = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)

    inp_r    = pygame.Rect(panel_r.x + pad,
                           panel_r.y + pad + title_h + text_h + 12,
                           panel_w - pad * 2, inp_h)

    gap      = 20
    total    = btn_w * 2 + gap
    ok_r     = pygame.Rect(cx - total // 2,              panel_r.bottom - pad - btn_h, btn_w, btn_h)
    cancel_r = pygame.Rect(cx - total // 2 + btn_w + gap, panel_r.bottom - pad - btn_h, btn_w, btn_h)

    text_buf = list(default_text)
    result   = [None]   # None = not decided yet; False = cancelled; str = submitted
    cursor_v = [True]
    cursor_t = [pygame.time.get_ticks()]
    clock    = pygame.time.Clock()

    while result[0] is None:
        now = pygame.time.get_ticks()
        if now - cursor_t[0] > 500:
            cursor_v[0] = not cursor_v[0]
            cursor_t[0] = now

        mx, my  = pygame.mouse.get_pos()
        surf.blit(snapshot, (0, 0))
        _overlay(surf)
        _draw_panel(surf, panel_r, title)

        ty = panel_r.y + pad + title_h
        for line in wrapped:
            ts = f_body.render(line, True, _TEXT)
            surf.blit(ts, (panel_r.x + pad, ty))
            ty += f_body.get_linesize() + 2

        # input box
        pygame.draw.rect(surf, _INPUT_ACT, inp_r, border_radius=5)
        pygame.draw.rect(surf, _BORDER,    inp_r, width=1, border_radius=5)
        disp = "".join(text_buf)
        # clip to fit
        while f_input.size(disp)[0] > inp_r.width - 12:
            disp = disp[1:]
        ts_i = f_input.render(disp, True, _TEXT)
        surf.blit(ts_i, (inp_r.x + 8, inp_r.y + (inp_h - ts_i.get_height()) // 2))
        if cursor_v[0]:
            cx_px = inp_r.x + 8 + f_input.size(disp)[0]
            pygame.draw.line(surf, _CURSOR,
                             (cx_px, inp_r.y + 6), (cx_px, inp_r.bottom - 6), 2)

        _draw_button(surf, ok_r,     ok_label,     ok_r.collidepoint(mx, my))
        _draw_button(surf, cancel_r, cancel_label, cancel_r.collidepoint(mx, my))
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ok_r.collidepoint(ev.pos):     result[0] = "".join(text_buf)
                if cancel_r.collidepoint(ev.pos): result[0] = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:  result[0] = "".join(text_buf)
                elif ev.key == pygame.K_ESCAPE: result[0] = False
                elif ev.key == pygame.K_BACKSPACE:
                    if text_buf: text_buf.pop()
                else:
                    ch = ev.unicode
                    if ch and ch.isprintable():
                        text_buf.append(ch)
        clock.tick(30)

    surf.blit(snapshot, (0, 0))
    pygame.display.flip()
    return None if result[0] is False else result[0]


def popup_get_file(msg: str, *, title: str = "", font=None,
                   initial_folder: Optional[str] = None) -> Optional[str]:
    """
    File-picker dialog.
    Shows a scrollable list of .pkl files from initial_folder (or CWD if None).
    Returns the full path string, or None if cancelled.
    """
    import os

    search_dir = initial_folder if (initial_folder and os.path.isdir(initial_folder)) else "."

    surf     = _get_display()
    snapshot = surf.copy()

    panel_w  = 560
    f_title  = _font(15, bold=True)
    f_body   = _font(15)
    f_item   = _font(15)
    pad      = 18
    list_h   = 260
    btn_h, btn_w = 42, 110
    item_h   = 32

    # collect pkl files
    try:
        files = sorted(
            f for f in os.listdir(search_dir) if f.lower().endswith(".pkl")
        )
    except Exception:
        files = []

    title_h  = (f_title.get_linesize() + 12)
    msg_h    = f_body.get_linesize() + 8
    panel_h  = pad + title_h + msg_h + list_h + pad + btn_h + pad

    cx2, cy2 = surf.get_width() // 2, surf.get_height() // 2
    panel_r  = pygame.Rect(cx2 - panel_w // 2, cy2 - panel_h // 2, panel_w, panel_h)
    list_r   = pygame.Rect(panel_r.x + pad,
                           panel_r.y + pad + title_h + msg_h,
                           panel_w - pad * 2, list_h)

    gap      = 20
    total    = btn_w * 2 + gap
    ok_r     = pygame.Rect(cx2 - total // 2,              panel_r.bottom - pad - btn_h, btn_w, btn_h)
    cancel_r = pygame.Rect(cx2 - total // 2 + btn_w + gap, panel_r.bottom - pad - btn_h, btn_w, btn_h)

    scroll   = 0
    selected = [None]   # index into files
    result   = [None]
    clock    = pygame.time.Clock()

    while result[0] is None:
        mx, my = pygame.mouse.get_pos()
        surf.blit(snapshot, (0, 0))
        _overlay(surf)
        _draw_panel(surf, panel_r, title or "Select a file")

        # message
        ms = f_body.render(msg, True, _TEXT_DIM)
        surf.blit(ms, (panel_r.x + pad, panel_r.y + pad + title_h))

        # list box background
        pygame.draw.rect(surf, _INPUT_BG, list_r, border_radius=6)
        pygame.draw.rect(surf, _BORDER,   list_r, width=1, border_radius=6)

        # clip list drawing
        surf.set_clip(list_r)
        vis = list_r.height // item_h
        for i, fname in enumerate(files[scroll: scroll + vis + 1]):
            iy  = list_r.y + i * item_h
            ir  = pygame.Rect(list_r.x, iy, list_r.width, item_h)
            idx = i + scroll
            if idx == selected[0]:
                pygame.draw.rect(surf, _BTN_NRM, ir)
            elif ir.collidepoint(mx, my):
                pygame.draw.rect(surf, (55, 50, 40), ir)
            ts = f_item.render(fname, True, _TEXT)
            surf.blit(ts, (ir.x + 8, iy + (item_h - ts.get_height()) // 2))
        surf.set_clip(None)

        if not files:
            ns = f_body.render("No .pkl files found in current directory", True, _TEXT_DIM)
            surf.blit(ns, ns.get_rect(center=list_r.center))

        _draw_button(surf, ok_r,     "Open",   ok_r.collidepoint(mx, my))
        _draw_button(surf, cancel_r, "Cancel", cancel_r.collidepoint(mx, my))
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if list_r.collidepoint(ev.pos):
                    idx = (ev.pos[1] - list_r.y) // item_h + scroll
                    if 0 <= idx < len(files):
                        selected[0] = idx
                if ok_r.collidepoint(ev.pos):
                    if selected[0] is not None:
                        result[0] = os.path.join(search_dir, files[selected[0]])
                    else:
                        result[0] = False
                if cancel_r.collidepoint(ev.pos):
                    result[0] = False
            if ev.type == pygame.MOUSEWHEEL:
                scroll = max(0, min(scroll - ev.y, max(0, len(files) - vis)))
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:    result[0] = False
                if ev.key == pygame.K_RETURN and selected[0] is not None:
                    result[0] = os.path.join(search_dir, files[selected[0]])
                if ev.key == pygame.K_DOWN and selected[0] is not None:
                    selected[0] = min(selected[0] + 1, len(files) - 1)
                if ev.key == pygame.K_UP and selected[0] is not None:
                    selected[0] = max(selected[0] - 1, 0)
        clock.tick(30)

    surf.blit(snapshot, (0, 0))
    pygame.display.flip()
    return None if result[0] is False else result[0]
