"""
sgf_to_training.py
==================
Converts 19x19 SGF game files into training samples written to saved_other_play.jsonl.
Run repeatedly to append more games to the same file.

Each sample is a tuple of:
  [0]  board_state_list  – flat float list encoding the 17-plane NN input
  [1]  policy_vector     – one-hot List[float] of length 362 (19*19 + pass)
  [2]  winner            – 1.0 if black won, -1.0 if white won

Usage:
    python sgf_to_training.py --sgf_dir ./sgf_files
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

BOARD_SIZE  = 19
OUTPUT_PATH = 'saved_other_play.jsonl'


# ── SGF parsing ───────────────────────────────────────────────────────────────

def _sgf_coord_to_rowcol(coord: str, board_size: int) -> Optional[Tuple[int, int]]:
    """Convert a two-character SGF coordinate (e.g. 'pd') to (row, col).
    Returns None for a pass (empty string or 'tt' on boards <=19)."""
    if not coord or coord.lower() == 'tt':
        return None  # pass
    col = ord(coord[0].lower()) - ord('a')
    row = ord(coord[1].lower()) - ord('a')
    if not (0 <= row < board_size and 0 <= col < board_size):
        return None  # out of bounds — treat as pass
    return (row, col)


def _extract_property(sgf_text: str, prop: str) -> Optional[str]:
    """Extract the first value of an SGF property, e.g. RE, SZ, KM."""
    match = re.search(rf'\b{prop}\[([^\]]*)\]', sgf_text)
    return match.group(1) if match else None


def _parse_moves(sgf_text: str) -> List[Tuple[str, str]]:
    """Return a list of (color, coord_str) for every B[] or W[] move in order."""
    return re.findall(r';([BW])\[([a-zA-Z]{0,2})\]', sgf_text)


def _determine_winner(sgf_text: str) -> Optional[float]:
    """Return 1.0 (black wins) or -1.0 (white wins), or None if unknown."""
    re_val = _extract_property(sgf_text, 'RE')
    if not re_val:
        return None
    re_val = re_val.strip().upper()
    if re_val.startswith('B'):
        return 1.0
    if re_val.startswith('W'):
        return -1.0
    return None  # jigo, void, unknown


# ── Board simulation ──────────────────────────────────────────────────────────

class _Board:
    """Minimal Go board: place stones, capture groups, no ko enforcement."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self, size: int):
        self.size = size
        self.grid = [[self.EMPTY] * size for _ in range(size)]

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        n = []
        if r > 0:          n.append((r - 1, c))
        if r < self.size - 1: n.append((r + 1, c))
        if c > 0:          n.append((r, c - 1))
        if c < self.size - 1: n.append((r, c + 1))
        return n

    def _get_group(self, r: int, c: int) -> Tuple[List[Tuple[int, int]], int]:
        """BFS to find group and count its liberties."""
        color = self.grid[r][c]
        visited = set()
        queue = [(r, c)]
        liberties = 0
        group = []
        while queue:
            cr, cc = queue.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            group.append((cr, cc))
            for nr, nc in self._neighbors(cr, cc):
                if self.grid[nr][nc] == self.EMPTY:
                    liberties += 1
                elif self.grid[nr][nc] == color and (nr, nc) not in visited:
                    queue.append((nr, nc))
        return group, liberties

    def _remove_group(self, group: List[Tuple[int, int]]) -> None:
        for r, c in group:
            self.grid[r][c] = self.EMPTY

    def place(self, r: int, c: int, color: int) -> bool:
        """Place a stone and capture. Returns False if illegal (occupied)."""
        if self.grid[r][c] != self.EMPTY:
            return False
        self.grid[r][c] = color
        opponent = self.WHITE if color == self.BLACK else self.BLACK
        # capture opponent groups with no liberties
        for nr, nc in self._neighbors(r, c):
            if self.grid[nr][nc] == opponent:
                group, libs = self._get_group(nr, nc)
                if libs == 0:
                    self._remove_group(group)
        return True

    def to_string(self, turn_color: int) -> str:
        """Encode board as '<turn_char><N*N chars>'.
        turn_char: '1'=black, '2'=white.  Cell chars: '0','1','2'."""
        cells = ''.join(str(self.grid[r][c])
                        for r in range(self.size)
                        for c in range(self.size))
        return str(turn_color) + cells

    def copy(self) -> '_Board':
        b = _Board(self.size)
        b.grid = [row[:] for row in self.grid]
        return b


# ── 17-plane input construction (mirrors generate_17_length in neuralnet.py) ──

def _build_17_plane_input(history: List[str], board_size: int) -> List[float]:
    """
    Converts a list of up to 17 board strings (most-recent LAST) into a
    flat 17 × board_size × board_size float array (row-major, returned as list).

    Mirrors generate_17_length() in neuralnet.py exactly.
    The last entry in `history` carries the colour-turn prefix char.
    """
    import copy as _copy

    arr = [[[0.0] * board_size for _ in range(board_size)] for _ in range(17)]

    history = _copy.copy(history)
    color_turn = int(history[-1][0]) if history[-1] else 1
    color_turn = 1 if color_turn == 1 else 2

    board_idx = 0

    while history:
        pop_board = history.pop(0)
        # strip the leading colour-turn prefix if present
        if len(pop_board) != board_size * board_size:
            pop_board = pop_board[1:]

        def set_black(idx_: int) -> None:
            for i in range(len(pop_board)):
                arr[idx_][i // board_size][i % board_size] = (
                    1.0 if int(pop_board[i]) == 1 else 0.0
                )

        def set_white(idx_: int) -> None:
            for i in range(len(pop_board)):
                arr[idx_][i // board_size][i % board_size] = (
                    1.0 if int(pop_board[i]) == 2 else 0.0
                )

        def set_end(idx_: int) -> None:
            for i in range(len(pop_board)):
                arr[idx_][i // board_size][i % board_size] = (
                    1.0 if int(pop_board[i]) == 1 else 0.0
                )

        if len(history) == 0:
            set_end(board_idx)
        elif color_turn == 1:
            if board_idx < 16:
                set_black(board_idx);     board_idx += 1
                set_white(board_idx);     board_idx += 1
        else:
            if board_idx < 16:
                set_white(board_idx);     board_idx += 1
                set_black(board_idx);     board_idx += 1

    # flatten to list (matches what json.dump produces from np arrays)
    flat = []
    for plane in arr:
        for row in plane:
            flat.extend(row)
    return flat


# ── Policy vector construction ────────────────────────────────────────────────

def _one_hot_policy(move: Optional[Tuple[int, int]], board_size: int) -> List[float]:
    """One-hot policy vector of length board_size²+1. Pass → last index."""
    n_moves = board_size * board_size + 1
    vec = [0.0] * n_moves
    if move is None:
        vec[-1] = 1.0          # pass
    else:
        row, col = move
        vec[row * board_size + col] = 1.0
    return vec


# ── Main conversion logic ─────────────────────────────────────────────────────

HISTORY_DEPTH = 8    # matches nn_input_generation in nnmcst.py ([-8:] window)

def sgf_to_samples(sgf_text: str) -> List[Tuple[List, List[float], float]]:
    """
    Parse one SGF string and return a list of training samples.
    Each sample: (board_state_list, policy_vector, value)

    value is +1.0 if the player to move at that board state won the game,
    -1.0 if they lost — matching the AlphaGo Zero convention.

    Returns [] if the game has no winner, is not 19x19, or has no moves.
    """
    sz_str = _extract_property(sgf_text, 'SZ')
    file_size = int(sz_str) if sz_str and sz_str.isdigit() else 19
    if file_size != BOARD_SIZE:
        return []

    winner = _determine_winner(sgf_text)
    if winner is None:
        return []

    moves = _parse_moves(sgf_text)
    if not moves:
        return []

    board = _Board(BOARD_SIZE)
    color_map = {'B': _Board.BLACK, 'W': _Board.WHITE}
    turn_map  = {'B': 1,            'W': 2}

    # seed history with 10 empty boards (mirrors NNBoard.play_game())
    empty_str = '0' * (BOARD_SIZE * BOARD_SIZE)
    history: List[str] = ['0' + empty_str] * 10

    samples = []

    for color_char, coord_str in moves:
        color    = color_map[color_char]
        turn_num = turn_map[color_char]
        move_rc  = _sgf_coord_to_rowcol(coord_str, BOARD_SIZE)

        # build training input BEFORE placing the stone (state seen by the player)
        recent_history = history[-HISTORY_DEPTH:]
        # reverse to newest-first, matching nn_input_generation in nnmcst.py
        # which does history[-8:].reverse() before passing to generate_17_length.
        # _build_17_plane_input pops from index 0, so index 0 must be the newest board.
        recent_history = list(reversed(recent_history))

        # Append a color sentinel matching nn_input_generation in nnmcst.py:
        # all '1's if black to move, all '0's if white to move.
        # set_end() in _build_17_plane_input encodes this as plane 16 = color indicator
        # (all 1.0 = black to move, all 0.0 = white to move), matching generate_17_length.
        if turn_num == 1:
            color_sentinel = '1' * BOARD_SIZE * BOARD_SIZE   # black to move
        else:
            color_sentinel = '0' * BOARD_SIZE * BOARD_SIZE   # white to move
        recent_history = recent_history + [color_sentinel]

        board_state_list = _build_17_plane_input(recent_history, BOARD_SIZE)
        policy_vector    = _one_hot_policy(move_rc, BOARD_SIZE)

        # AlphaGo Zero convention: value target = +1 if the player TO MOVE won,
        # -1 if the player to move lost.
        # winner is +1.0 (black won) or -1.0 (white won).
        # turn_num is 1 (black to move) or 2 (white to move).
        # If black won  and black is moving → +1.0  (mover wins)
        # If black won  and white is moving → -1.0  (mover loses)
        # If white won  and white is moving → +1.0  (mover wins)
        # If white won  and black is moving → -1.0  (mover loses)
        black_won    = (winner == 1.0)
        black_moving = (turn_num == 1)
        perspective  = 1.0 if (black_won == black_moving) else -1.0

        samples.append((board_state_list, policy_vector, perspective))

        if move_rc is not None:
            board.place(move_rc[0], move_rc[1], color)

        next_turn = 2 if color == _Board.BLACK else 1
        history.append(board.to_string(next_turn))

    return samples


def convert_sgf_dir(sgf_dir: str, verbose: bool = True) -> int:
    """
    Convert all .sgf files in `sgf_dir` and append results to saved_other_play.jsonl.
    Uses JSONL format (one sample per line) so the file is never fully loaded into
    memory — each game's samples are written immediately after conversion.
    Always appends — run multiple times to accumulate data.

    Returns total number of samples now in the output file.
    """
    # Count existing samples without loading them
    existing_count = 0
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    existing_count += 1
        if verbose:
            print(f"Found {existing_count} existing samples in {OUTPUT_PATH}")

    sgf_files = []
    for root, _, files in os.walk(sgf_dir):
        for fname in files:
            if fname.lower().endswith('.sgf'):
                sgf_files.append(os.path.join(root, fname))

    if verbose:
        print(f"Found {len(sgf_files)} SGF file(s) in '{sgf_dir}'")

    new_count          = 0
    skipped_no_result  = 0
    skipped_wrong_size = 0
    skipped_error      = 0
    WRITE_BATCH = 100  # flush to disk every N games

    # Open in append mode — stream batches of games directly to disk
    with open(OUTPUT_PATH, 'a') as out_f:
        batch = []
        for i, fpath in enumerate(sgf_files):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
            except Exception as e:
                if verbose:
                    print(f"  [SKIP] {fpath}: read error — {e}")
                skipped_error += 1
                continue

            sz_str = _extract_property(text, 'SZ')
            file_size = int(sz_str) if sz_str and sz_str.isdigit() else 19
            if file_size != BOARD_SIZE:
                skipped_wrong_size += 1
                continue

            samples = sgf_to_samples(text)
            if not samples:
                skipped_no_result += 1
                if verbose:
                    print(f"  [SKIP] {fpath}: no result or no moves")
                continue

            batch.extend(samples)
            new_count += len(samples)

            # flush batch to disk every WRITE_BATCH games
            if (i + 1) % WRITE_BATCH == 0:
                for sample in batch:
                    out_f.write(json.dumps(sample) + '\n')
                out_f.flush()
                batch = []
                if verbose:
                    print(f"  Processed {i+1}/{len(sgf_files)} files — "
                          f"{new_count} new samples written to disk...")

        # flush any remaining samples
        for sample in batch:
            out_f.write(json.dumps(sample) + '\n')

    total = existing_count + new_count
    if verbose:
        print(f"\n── Summary ──────────────────────────────────")
        print(f"  SGF files found:        {len(sgf_files)}")
        print(f"  Skipped (wrong size):   {skipped_wrong_size}")
        print(f"  Skipped (no result):    {skipped_no_result}")
        print(f"  Skipped (read error):   {skipped_error}")
        print(f"  New samples written:    {new_count}")
        print(f"  Existing samples kept:  {existing_count}")
        print(f"  Total in output:        {total}")
        print(f"  Output written to:      {OUTPUT_PATH}")

    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert 19x19 SGF files to saved_other_play.json training data."
    )
    p.add_argument(
        '--sgf_dir', required=True,
        help="Directory containing .sgf files (searched recursively)."
    )
    p.add_argument(
        '--quiet', action='store_true',
        help="Suppress progress output."
    )
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.sgf_dir):
        print(f"ERROR: '{args.sgf_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    convert_sgf_dir(sgf_dir=args.sgf_dir, verbose=not args.quiet)


if __name__ == '__main__':
    main()
