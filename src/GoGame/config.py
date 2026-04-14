# ── colour constants ──────────────────────────────────────────────────────────
rgb_black    = (0, 0, 0)
rgb_white    = (255, 255, 255)
rgb_grey     = (120, 120, 120)
rgb_green    = (50, 205, 50)
rgb_red      = (195, 33, 72)
rgb_peach    = (239, 159, 118)   
rgb_lavender = (186, 187, 241)   

# ── AI configuration ─────────────────────────────────────────────────────────
# Change these two values to adjust board size and MCTS strength.
# AI_BOARD_SIZE: 9, 13, or 19
# MCTS_ITERATIONS: 25 = fast/weak, 200 = good balance with GPU, 800 = strong/slow
AI_BOARD_SIZE        = 19
MCTS_ITERATIONS      = 200 # used during self-play training
PLAY_MCTS_ITERATIONS = 25   # used when playing against a human

# Set to True to allow the AI to resign during self-play when its position is
# hopeless. Requires two consecutive turns below the value threshold.
# Disable this during early training when the value head is not yet calibrated —
# premature resignations produce incorrectly labelled training samples.
ALLOW_RESIGNATION = False
RESIGNATION_THRESHOLD = -0.99  # value below which the AI considers resigning (-1.0 to 0.0)

# ── Release mode ──────────────────────────────────────────────────────────────
# Set to True when building the distributable exe.
# Hides AI training, self-play, SGF import, and SGF training from the menu.
RELEASE_MODE = False
