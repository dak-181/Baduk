import GoGame.neuralnetboard as nn
import numpy as np
from typing import List, Optional
import sys
import tensorflow.keras as keras
import GoGame.config as cf
np.set_printoptions(threshold=sys.maxsize)

# derived from config — number of board intersections + 1 pass move
_BOARD  = cf.AI_BOARD_SIZE
_MOVES  = _BOARD * _BOARD + 1   # e.g. 362 for 19x19, 82 for 9x9


def nn_model(board_size: int = _BOARD):
    """
    Builds the AlphaGo Zero model for the configured board size.
    Policy head outputs board_size² + 1 values (all moves + pass).
    Loads saved weights if available.
    """
    moves   = board_size * board_size + 1
    shapez  = (17, board_size, board_size)
    input_layer  = keras.layers.Input(shape=shapez)
    conv_output  = nn_model_conv_layer(input_layer)
    res_output   = nn_model_res_layer(conv_output)
    for _ in range(9):
        res_output = nn_model_res_layer(res_output)
    policy_output = nn_model_policy_head(res_output, moves)
    value_output  = nn_model_value_head(res_output)
    model = keras.models.Model(
        inputs=input_layer,
        outputs={'dense_2': value_output, 'softmax': policy_output}
    )
    load_model_weights(model)
    return model


def nn_model_conv_layer(input_array):
    conv1    = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(input_array)
    b_norm_1 = keras.layers.BatchNormalization()(conv1)
    return keras.layers.ReLU()(b_norm_1)


def nn_model_res_layer(input_array):
    conv1    = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(input_array)
    b_norm_1 = keras.layers.BatchNormalization()(conv1)
    relu_1   = keras.layers.ReLU()(b_norm_1)
    conv2    = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(relu_1)
    b_norm_2 = keras.layers.BatchNormalization()(conv2)
    added    = keras.layers.Add()([input_array, b_norm_2])
    return keras.layers.ReLU()(added)


def nn_model_policy_head(input_array, moves: int = _MOVES):
    """Policy head: outputs softmax over all board moves + pass."""
    conv1    = keras.layers.Conv2D(2, (1, 1), strides=(1, 1), padding='same')(input_array)
    b_norm_1 = keras.layers.BatchNormalization()(conv1)
    relu_1   = keras.layers.ReLU()(b_norm_1)
    flatten  = keras.layers.Flatten()(relu_1)
    dense    = keras.layers.Dense(moves)(flatten)
    return keras.layers.Softmax()(dense)


def nn_model_value_head(input_array):
    conv1    = keras.layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(input_array)
    b_norm_1 = keras.layers.BatchNormalization()(conv1)
    relu_1   = keras.layers.ReLU()(b_norm_1)
    flatten  = keras.layers.Flatten()(relu_1)
    dense1   = keras.layers.Dense(256)(flatten)
    relu_2   = keras.layers.ReLU()(dense1)
    return keras.layers.Dense(1, activation='tanh')(relu_2)


def training_cycle(length: int, state: Optional[dict] = None):
    """
    Plays `length` self-play games to generate training data.
    Saves results to saved_self_play.json after each game.

    state: optional dict shared with the UI thread for progress reporting.
           Keys used: 'log' (list of str), 'game' (int), 'cancelled' (bool).
    """
    import time

    def log(msg: str):
        print(msg)
        if state is not None:
            state.setdefault("log", []).append(msg)

    sum_val    = 0
    start_time = time.time()
    nn_mod     = nn_model()
    log(f"Starting {length} self-play games...")

    for i in range(length):
        if state and state.get("cancelled"):
            log("Cancelled by user.")
            break

        log(f"Game {i+1}/{length} starting... (board: {_BOARD}x{_BOARD})")
        result = nn.initializing_game(nn_mod, nn_mod, _BOARD, True)
        if result == 1:
            sum_val += 1
        if state is not None:
            state["game"] = i + 1
        elapsed = time.time() - start_time
        log(f"Game {i+1}/{length} done — black wins: {sum_val}  ({elapsed:.0f}s elapsed)")

    total = time.time() - start_time
    log(f"Self-play complete: {sum_val}/{length} black wins in {total:.1f}s")
    log("Training data saved to saved_self_play.json")
    log("Run 'AI Training' to train the model on this data.")


def neural_net_calcuation(input_boards: List[str], board_size: int, input_nn):
    """Runs the neural net on the given board history. Returns (value, policy)."""
    input_array = generate_17_length(input_boards, board_size)
    input_array = np.expand_dims(input_array, axis=0)
    output        = input_nn.predict(input_array, verbose=0)
    value_output  = float(output['dense_2'][0][0])
    policy_output = output['softmax']
    return (value_output, policy_output)


def generate_17_length(input_boards: List[str], board_size: int):
    """Converts board history into the 17-plane numpy array for AlphaGo Zero."""
    input_array = np.zeros((17, board_size, board_size), dtype=np.float32)
    board_idx   = 0
    color_turn  = int(input_boards[-1][0]) if input_boards[-1] else 1
    color_turn  = 1 if color_turn == 1 else 2

    while input_boards:
        pop_board = input_boards.pop(0)
        if len(pop_board) != board_size * board_size:
            pop_board = pop_board[1:]

        if color_turn == 1 and len(input_boards) != 0:
            helper_black(input_array, pop_board, board_size, board_idx); board_idx += 1
            helper_white(input_array, pop_board, board_size, board_idx); board_idx += 1
        elif color_turn == 2 and len(input_boards) != 0:
            helper_white(input_array, pop_board, board_size, board_idx); board_idx += 1
            helper_black(input_array, pop_board, board_size, board_idx); board_idx += 1

        if len(input_boards) == 0:
            helper_end(input_array, pop_board, board_size, board_idx)
    return input_array


def helper_black(input_array, pop_board, board_size, board_idx):
    for idx in range(len(pop_board)):
        input_array[board_idx][idx // board_size][idx % board_size] = (
            1 if int(pop_board[idx]) == 1 else 0
        )


def helper_white(input_array, pop_board, board_size, board_idx):
    for idx in range(len(pop_board)):
        input_array[board_idx][idx // board_size][idx % board_size] = (
            1 if int(pop_board[idx]) == 2 else 0
        )


def helper_end(input_array, pop_board, board_size, board_idx):
    for idx in range(len(pop_board)):
        input_array[board_idx][idx // board_size][idx % board_size] = (
            1 if int(pop_board[idx]) == 1 else 0
        )


def save_model_weights(model, filename="model_weights.h5"):
    model.save_weights(filename)


def load_model_weights(model, filename="model_weights.h5"):
    import os
    if os.path.exists(filename):
        model.load_weights(filename)
    else:
        print(f"No saved weights at '{filename}' — starting with random weights.")


def loading_file_for_training(epochs: int = 10, size_of_batch: int = 32,
                               state: Optional[dict] = None,
                               run_evaluation: bool = False,
                               eval_games: int = 20):
    """
    Trains the model on self-play data from saved_self_play.json and saves
    the updated weights immediately.

    Parameters:
        epochs:         number of training epochs (default 10)
        size_of_batch:  minibatch size (default 32)
        state:          optional progress dict shared with the UI thread
        run_evaluation: if True, play eval_games after training to measure
                        improvement before saving. Off by default — only
                        useful once you have substantial training data.
        eval_games:     number of evaluation games when run_evaluation=True
    """
    import json, random, os

    def log(msg: str):
        print(msg)
        if state is not None:
            state.setdefault("log", []).append(msg)

    if not os.path.exists("saved_self_play.json"):
        log("No training data found. Run 'AI Self Play' first.")
        return

    with open("saved_self_play.json", "r") as jfile:
        dataset = json.load(jfile)

    if len(dataset) < size_of_batch:
        log(f"Not enough data ({len(dataset)} samples). Run more self-play games.")
        return

    log(f"Loaded {len(dataset)} training samples.")
    model       = nn_model()
    value_loss  = keras.losses.MeanSquaredError()
    policy_loss = keras.losses.CategoricalCrossentropy()
    sample_size = max(size_of_batch, len(dataset) // 4)

    for epoch in range(epochs):
        if state and state.get("cancelled"):
            log("Training cancelled by user.")
            return
        selected = random.sample(dataset, min(sample_size, len(dataset)))
        inputs   = np.array([np.asarray(s[0], dtype=np.float32) for s in selected])
        outputs  = {
            'dense_2': np.array([s[2] for s in selected]),
            'softmax': np.array([s[1] for s in selected])
        }
        model.compile(
            optimizer='adam',
            loss={'dense_2': value_loss, 'softmax': policy_loss},
            metrics={'softmax': ['accuracy']}
        )
        model.fit(inputs, outputs, epochs=1, batch_size=size_of_batch, verbose=1)
        log(f"Epoch {epoch+1}/{epochs} complete.")

    if state and state.get("cancelled"):
        log("Training cancelled before saving.")
        return

    # ── optional evaluation ───────────────────────────────────────────────
    if run_evaluation:
        log(f"Evaluating new model ({eval_games} games)...")
        nn_new  = model
        nn_old  = nn_model()   # loads current saved weights as baseline
        sum_val = 0
        for i in range(eval_games):
            if state and state.get("cancelled"):
                log("Evaluation cancelled.")
                return
            result = nn.initializing_game(nn_new, nn_old, _BOARD, True)
            if result == 1:
                sum_val += 1
            log(f"  Eval game {i+1}/{eval_games} — new model wins: {sum_val}")
        win_rate = sum_val / eval_games
        log(f"New model win rate: {win_rate:.1%}")
        if win_rate >= 0.55:
            save_model_weights(model)
            log("New weights saved — model improved!")
        else:
            log("Model did not reach 55% threshold — keeping current weights.")
    else:
        # just save — training always improves the model when data is good
        save_model_weights(model)
        log("Weights saved to model_weights.h5")
