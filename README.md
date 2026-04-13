# Go Go Go

A Python implementation of the board game Go (Baduk/Weiqi) with a pygame GUI, a random bot opponent, and an AlphaGo Zero-style neural network AI that can be trained via self-play or imported SGF game records.

Originally forked from [EGPeat/GoGame](https://github.com/EGPeat/GoGame) and modernised by replacing deprecated PySimpleGUI dependencies with a pure pygame interface.

---

## Requirements

- Python 3.12
- pygame
- numpy
- TensorFlow
- Keras 
- CUDA

GPU recommended — tested on RTX 3090 Ti via WSL2 + CUDA 12.3 and TensorFlow 2.16. You should configure the proper version of CUDA and TensorFlow for your GPU

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Game

```bash
python run.py
```

---

## Main Menu

| Button | Description |
|---|---|
| **Load Game** | Load a previously saved game from the `pklfiles/` folder |
| **New Custom Game** | Start a new game with custom board size, player names, komi, and handicap |
| **New Default Game** | Start a 9×9 game with default settings (7.5 komi, Player 1 vs Player 2) |
| **Play Against AI** | Play against either the random bot or a trained neural network model |
| **AI SelfPlay** | Run AI vs AI self-play games to generate training data |
| **AI Training** | Train the main AI model on self-play data |
| **Import SGF Files** | Convert SGF game records into training data |
| **Train SGF Model** | Train a separate AI model on imported SGF data |
| **Exit Game** | Quit the application |

---

## Playing a Game

### Controls

The button bar at the top of the board provides the following actions:

| Button | Description |
|---|---|
| **Pass Turn** | Pass your turn |
| **Undo Turn** | Undo the last move |
| **Resume Game** | Return to play from the scoring phase |
| **Save Game** | Save the current game to a `.pkl` file |
| **Exit To Menu** | Return to the main menu |

### Scoring

When both players pass consecutively the game enters scoring mode. The **Pass Turn** button changes to **Accept**.

1. Click any stone group you believe is dead — it will be faded to indicate it is marked for removal
2. Click a faded stone group again to restore it as alive
3. Click **Accept** to finalise dead stone removal and calculate the score

The board will display territory markers (dark squares for black territory, light squares for white territory) on top of the final position before showing the score breakdown in the sidebar.

**Resume Game** returns you to play if you disagree with the dead stone marking.

### Saving and Loading

- Games are saved as `.pkl` files in a `pklfiles/` subfolder
- Click **Save Game** at any point during play
- Click **Load Game** from the main menu to resume a saved game

---

## Playing Against AI

Click **Play Against AI** from the main menu to choose your opponent:

- **Random Bot** — plays random legal moves, good for testing
- Any `.h5` weights file found in the working directory — plays using the neural network loaded from that file

You play as **Black**. The AI plays as **White**.

---

## AI Tools

### AI SelfPlay

Generates training data by having the neural network play against itself.

1. Click **AI SelfPlay**
2. Enter the number of games to play (default: 5)
3. Progress is shown on screen — press **Escape** to stop immediately
4. Data is saved to `saved_self_play.jsonl` and can be used with **AI Training**

### AI Training

Trains the main neural network model on self-play data.

1. Run **AI SelfPlay** first to generate `saved_self_play.jsonl`
2. Click **AI Training**
3. Press **Escape** to stop immediately at any point
4. Weights are saved to `model.weights.h5`

### Importing SGF Files

Converts existing SGF game records into training data for the neural network.

1. Create an `sgf/` folder in the project root (next to `run.py`)
2. Place your `.sgf` files inside it (subfolders are searched recursively)
3. Click **Import SGF Files** from the main menu
4. Only 19×19 games with a recorded result are converted
5. Data is saved to `saved_other_play.jsonl`

### Train SGF Model

Trains a separate neural network model on SGF-imported data, completely independent of the self-play model.

1. Run **Import SGF Files** first to generate `saved_other_play.jsonl`
2. Click **Train SGF Model**
3. Press **Escape** to stop immediately at any point
4. Weights are saved to `other_play.weights.h5` (rename this after creation to save multiple models i.e. shusaku.weights.h5)
5. This model will appear in the **Play Against AI** opponent picker

---

## AI Configuration

Three settings in `src/GoGame/config.py` control the AI:

```python
AI_BOARD_SIZE   = 19   # 9, 13, or 19
MCTS_ITERATIONS = 200   # Used during self-play training
PLAY_MCTS_ITERATIONS = 25   # used when playing against a human
```
Iterations: 25 = fast/weak · 200 = strong · 800 = very strong/slow.

> **Note:** Changing `AI_BOARD_SIZE` requires deleting any existing `.h5` weight files since the model architecture changes with board size.

---

## Project Structure

```
run.py                          ← entry point
src/GoGame/
    main.py                     ← main menu loop
    config.py                   ← colours, AI board size, MCTS iterations
    goclasses.py                ← GoBoard, game logic
    uifunctions.py              ← pygame rendering and window layout
    pygame_ui.py                ← popup/dialog system
    stone_renderer.py           ← Phong-shaded stone rendering
    game_initialization.py      ← board setup and player configuration
    player.py                   ← Player class
    turn_options.py             ← button event handling
    saving_loading.py           ← pickle save/load
    remove_dead.py              ← dead stone marking during scoring
    scoringboard.py             ← territory scoring
    undoing.py                  ← undo logic
    handicap.py                 ← handicap placement
    botnormalgo.py              ← random bot and neural net bot
    neuralnet.py                ← AlphaGo Zero model, training
    neuralnetboard.py           ← headless AI self-play board
    nnmcst.py                   ← neural net MCTS
    mcst.py                     ← base MCTS classes
    sgf_to_training.py          ← SGF to training data converter
pklfiles/                       ← saved games
sgf/                            ← place SGF files here for import
model.weights.h5                ← self-play model weights
other_play.weights.h5           ← SGF-trained model weights
saved_self_play.jsonl           ← self-play training data
saved_other_play.jsonl          ← SGF training data
```
