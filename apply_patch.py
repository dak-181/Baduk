#!/usr/bin/env python3
"""
apply_patch.py
Run this from the repo root:
    python apply_patch.py

It patches src/GoGame/goclasses.py in-place, removing all PySimpleGUI
references and replacing the window.read() event loop with pygame.
"""

import re, shutil, sys
from pathlib import Path

TARGET = Path("src/GoGame/goclasses.py")

if not TARGET.exists():
    sys.exit(f"ERROR: {TARGET} not found. Run from repo root.")

# backup
shutil.copy(TARGET, TARGET.with_suffix(".py.bak"))
print(f"Backed up to {TARGET.with_suffix('.py.bak')}")

src = TARGET.read_text()

# ── CHANGE 1: remove PySimpleGUI import ─────────────────────────────────────
src = re.sub(r"^import PySimpleGUI as sg\n", "", src, flags=re.MULTILINE)
print("✓ Removed: import PySimpleGUI as sg")

# ── CHANGE 2: fix window type annotation ────────────────────────────────────
src = src.replace(
    "self.window: Union[sg.Window, None] = None",
    "self.window: None = None  # no sg.Window; managed by uifunctions"
)
print("✓ Fixed: self.window type annotation")

# ── CHANGE 3: replace read_window() ─────────────────────────────────────────
OLD_READ = '''\
    def read_window(self) -> Tuple[str, dict]:
        \'\'\'Reads the window for any input values from clicks, and returns those values.\'\'\'
        event, values = self.window.read()
        return event, values'''

NEW_READ = '''\
    def read_window(self) -> Tuple[str, dict]:
        \'\'\'
        Pygame event loop replacement for sg.Window.read().
        Returns (event_name, values) where:
          - event_name is a button label string, None (WIN_CLOSED), or "-GRAPH-"
          - values is {"-GRAPH-": (x, y)} when a board click occurs
        \'\'\'
        import pygame
        import GoGame.uifunctions as _ui
        clock = pygame.time.Clock()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None, {}

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos

                    # check button bar first
                    for label, rect in self.btn_rects.items():
                        if rect.collidepoint(mx, my):
                            return label, {}

                    # click was in the board area (below the button bar)
                    board_y = my - _ui.BTN_BAR_H
                    if board_y >= 0:
                        return "-GRAPH-", {"-GRAPH-": (mx, board_y)}

                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return None, {}

            # keep rendering while waiting for input
            _ui.refresh_board_pygame(self)
            clock.tick(60)'''

if OLD_READ in src:
    src = src.replace(OLD_READ, NEW_READ)
    print("✓ Replaced: read_window()")
else:
    print("⚠ WARNING: Could not find read_window() to replace — check manually")

# ── CHANGE 4: remove scoring_block window["Res"].update() ───────────────────
OLD_RES = '''\
            if self.mode_change:
                if not self.mode == "Scoring":
                    self.window["Res"].update("Quit Program")'''
if OLD_RES in src:
    src = src.replace(OLD_RES, "")
    print('✓ Removed: self.window["Res"].update(...) in scoring_block()')
else:
    print('⚠ WARNING: Could not find window["Res"].update — check manually')

# ── CHANGE 5: fix play_game_view_endgame() ──────────────────────────────────
OLD_VIEW = '''\
    def play_game_view_endgame(self) -> None:
        \'\'\'Allows the user to view a completed game\'\'\'
        ui.refresh_board_pygame(self)
        event, _ = self.read_window()
        if event == "Exit Game":
            from GoGame.main import play_game_main
            ui.close_window(self)
            play_game_main()
            quit()
        elif event == sg.WIN_CLOSED:
            quit()'''

NEW_VIEW = '''\
    def play_game_view_endgame(self) -> None:
        \'\'\'Allows the user to view a completed game\'\'\'
        ui.refresh_board_pygame(self)
        event, _ = self.read_window()
        if event in ("Exit To Menu", None):
            from GoGame.main import play_game_main
            ui.close_window(self)
            play_game_main()
            quit()'''

if OLD_VIEW in src:
    src = src.replace(OLD_VIEW, NEW_VIEW)
    print("✓ Fixed: play_game_view_endgame()")
else:
    print("⚠ WARNING: Could not find play_game_view_endgame() — check manually")

# ── CHANGE 6: fix play_turn() early-return list ─────────────────────────────
OLD_TURN = '                if event == "Pass Turn" or event == "Res" or event == "Undo Turn":'
NEW_TURN = '                if event in ("Pass Turn", "Quit Program", "Undo Turn", "Exit To Menu", None):'

if OLD_TURN in src:
    src = src.replace(OLD_TURN, NEW_TURN)
    print("✓ Fixed: play_turn() early-return condition")
else:
    print("⚠ WARNING: Could not find play_turn() early-return — check manually")

# ── write result ─────────────────────────────────────────────────────────────
TARGET.write_text(src)
print(f"\n✅ Patch applied to {TARGET}")
print("   Original backed up as goclasses.py.bak")
print("\nNext: python run.py")
