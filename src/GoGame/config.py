global rgb_black
global rgb_white
global rgb_grey
global rgb_green
global rgb_red
global rgb_peach
global rgb_lavender
rgb_black = (0, 0, 0)
rgb_white = (255, 255, 255)
rgb_grey = (120, 120, 120)
rgb_green = (50, 205, 50)
rgb_red = (195, 33, 72)
rgb_peach = (239, 159, 118)  # represents dead pieces
rgb_lavender = (186, 187, 241)

# ── AI configuration ─────────────────────────────────────────────────────────
# Change these two values to adjust board size and MCTS strength.
# AI_BOARD_SIZE: 9, 13, or 19
# MCTS_ITERATIONS: 25 = fast/weak, 200 = good balance with GPU, 800 = strong/slow
AI_BOARD_SIZE    = 19
MCTS_ITERATIONS  = 100
