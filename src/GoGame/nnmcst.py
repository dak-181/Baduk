from GoGame.goclasses import BoardNode, BoardString
from typing import Tuple, List, Set, Union, Type, Dict, FrozenSet, Literal
from GoGame.player import Player
from GoGame.mcst import MCSTNode, MCST
import GoGame.config as cf
import math
from GoGame.neuralnet import neural_net_calcuation, generate_17_length
import copy
from numpy import argmax

_BOARD = cf.AI_BOARD_SIZE
_PASS  = _BOARD * _BOARD       # index of the pass move
_MOVES = _BOARD * _BOARD + 1   # total moves including pass


class NNMCSTNode(MCSTNode):
    def __init__(self, turn_person: Tuple[Player, Player], training_info: List[str], prob,
                 board_list=None, killed_last: Union[Set[None], Set[BoardNode]] = None,
                 placement_location: Tuple[Union[str, Tuple[int, int]], int, Tuple[int, int, int]] = ((-1, -1), -1, -1),
                 parent: Union[None, Type['NNMCSTNode']] = None) -> None:
        """
        Initializes an MCSTNode instance representing a Monte Carlo Search Tree node.

        Parameters:
            turn_person (Tuple[Player, Player]): Tuple of two Player instances representing whose_turn and not_whose_turn
            training_info (List[str]): list of strings representing boardstates in previous turns.
            prob: probability of being chosen by the parent NNMCSTNode.
            board_list (List[str]): List of strings representing the current state of the board.
            killed_last Set[BoardNode] or Empty Set: Set of BoardNode instances representing pieces killed in the last turn.
            placement_location: Tuple containing information about the placement location (move, row, col, and stone color).
            parent (MCSTNode): Reference to the parent MCSTNode instance, if one exists.

        Attributes:
            placement_choice: The chosen placement location (move) for this node.
            choice_info: Information about the placement location.
            board_list (List[str]): List of strings representing the current state of the board.
            parent (MCSTNode): Reference to the parent MCSTNode instance.
            children (List[MCSTNode]): List of child MCSTNode instances.
            move_choices (Dict[str, BoardNode]): Dictionary mapping moves to BoardNode instances.
            prior_probability (float): Probability of this node being chosen by its parent node.
            number_times_chosen (int): Number of times this node or it's children were chosen.
            total_v_children (float): Total value output of this node and all of it's children.
            mean_value_v: total_v_children divided by the number of times this node and it's children were chosen.
            ai_training_info_node (List[str]): training_info parameter.
            killed_last_turn (Union[Set[None], Set[BoardNode]]): Set of BoardNode instances killed in the last turn.
            child_killed_last (Union[Set[BoardNode], Set[None]]): Set of BoardNode instances killed by child nodes.
            visit_kill (Set[BoardNode]): Set of BoardNode instances representing visited and killed stones.
            whose_turn (Player): Player instance representing the current player's turn.
            not_whose_turn (Player): Player instance representing the not current player's turn.
            cache_hash (str): Hash value representing the state of the MCSTNode for caching purposes.

        Note:
            The default values for parameters are provided for optional attributes.
        """
        self.placement_choice = placement_location[0]
        self.choice_info = placement_location
        self.board_list: List[str] = board_list
        self.parent: Union[None, Type['NNMCSTNode']] = parent
        self.children: List[NNMCSTNode] = []
        self.move_choices: Dict[str, BoardNode] = dict()

        self.prior_probability: float = prob
        self.number_times_chosen: int = 0
        self.total_v_children: float = 0
        self.mean_value_v: float = 0
        self.ai_training_info_node = training_info

        self.killed_last_turn: Union[Set[None], Set[BoardNode]] = killed_last if killed_last is not None else set()
        self.child_killed_last: Union[Set[BoardNode], Set[None]] = set()
        self.visit_kill: Set[BoardNode] = set()
        self.whose_turn: Player = turn_person[0]
        self.not_whose_turn: Player = turn_person[1]
        self.mcstnode_init()
        self.cache_hash: str = self.generate_cache()


class NNMCST(MCST):
    def __init__(self, board: List[List[BoardNode]], training_info: List[str], white_board: str, black_board: str,
                 iterations: int, turn_person: Tuple[Player, Player], nn, turnnum: int,
                 training: bool = False) -> None:
        """
        Initializes an MCST instance representing a Monte Carlo Search Tree for game tree traversal and decision-making.

        Parameters:
            board (List[List[BoardNode]]): 2D list representing the current state of the board with BoardNode instances.
            training_info (List[str]): list of strings representing boardstates in previous turns.
            white_board (str): A string representing the entire board being of the value held by the white player
            black_board (str): A string representing the entire board being of the value held by the black board
            iterations (int): Number of iterations for Monte Carlo Tree Search.
            turn_person (Tuple[Player, Player]): Tuple of two Player instances representing the current and not current player.
            nn: neural net to be used to improve the MCST.
            turnnum (int): the turn number
            training (bool): if True, Dirichlet noise is added to root priors for exploration (self-play only).

        Attributes:
            board: 2D list representing the current state of the board with BoardNode instances.
            board_BoardString: BoardString representing the entire board.
            training_info (List[str]): list of strings representing boardstates in previous turns.
            white_board (str): A string representing the entire board being of the value held by the white player
            black_board (str): A string representing the entire board being of the value held by the black player
            cache: Dictionary caching frozen sets of BoardNode instances for each unique board state.
            win_cache: Dictionary caching win statistics for each unique board state.
            cache_hash: Hash value representing the state of the MCST for caching purposes.
            iteration_number: Number of iterations for Monte Carlo Tree Search.
            max_simulation_depth: Maximum simulation depth for Monte Carlo Tree Search.
            root: Root node of the Monte Carlo Search Tree.
            nn: neural net to be used
            nn_bad: copy of neural net, or a more outdated version.
            temp: temperature to be used by some functions. Either 1 or 0.1
            training: whether to apply Dirichlet noise to root priors.
        """

        self.board = board
        self.board_BoardString = None
        self.ai_training_info = copy.deepcopy(training_info)
        self.ai_white_board = white_board
        self.ai_black_board = black_board

        self.cache: Dict[str, FrozenSet[BoardNode]] = {}
        self.win_cache: Dict[str, Tuple[int, int]] = {}
        self.cache_hash: str = None
        self.secondary_init()
        self.iteration_number: int = iterations
        board_list_for_root = self.make_board_string()
        self.root: NNMCSTNode = NNMCSTNode(turn_person, self.ai_training_info, 1, board_list_for_root,
                                           placement_location=("Root", -1, -1))
        self.neural_net_inst = nn
        self.turn_num = turnnum
        self.training = training
        if turnnum <= 30:
            self.temp = 1
        else:
            self.temp = 0.1

    def secondary_init(self) -> None:
        '''Helper function for init.'''
        temp_set: Set[BoardNode] = set()
        for idx_1 in range(_BOARD):
            for idx_2 in range(_BOARD):
                temp_set.add(self.board[idx_1][idx_2])
        self.board_BoardString = BoardString(cf.rgb_grey, temp_set)

    def best_child_finder(self, node: NNMCSTNode) -> NNMCSTNode:
        '''Finds the best child of the given node, and then returns it.'''
        current_best_val = float('-inf')
        current_best_child: NNMCSTNode = None

        for child in node.children:
            child_val = self.get_UCB_score(child)
            if child_val > current_best_val:
                current_best_val = child_val
                current_best_child = child
        return current_best_child

    def get_UCB_score(self, child: NNMCSTNode) -> float:
        '''Calculate the Upper Confidence Bound (UCB) score for a given child node.
        child.mean_value_v is stored from the CHILD's player perspective (opponent of parent).
        We negate it so the parent always selects the child that is best from the parent's
        own perspective — i.e. worst for the opponent. This is the standard negamax convention.'''
        explor_weight = 1.4
        t_node = child
        if t_node.parent:
            t_node = t_node.parent
        penalty_term_inner_upper = 0
        for sibling in t_node.children:
            penalty_term_inner_upper += sibling.number_times_chosen

        penalty_term_inner_upper = math.sqrt(penalty_term_inner_upper)
        penalty_term_inner = penalty_term_inner_upper / (1 + child.number_times_chosen)
        penalty_term = explor_weight * child.prior_probability * penalty_term_inner
        ucb_value = -child.mean_value_v + penalty_term  # negate: child stores value from child's perspective
        return ucb_value

    def get_deep_score(self, child: NNMCSTNode) -> float:
        '''Calculate the score used by Deep Mind to choose the final output for a given child node, and returns it as a float.'''
        t_node = child
        if t_node.parent:
            t_node = t_node.parent
        lower_value = 0
        for sibling in t_node.children:
            lower_value += math.pow(sibling.number_times_chosen, (1 / self.temp))

        if lower_value == 0:
            return 0.0
        upper_value = math.pow(child.number_times_chosen, (1 / self.temp))
        return (upper_value / lower_value)

    def backpropagate(self, node: NNMCSTNode, value_output: float) -> None:
        '''Backpropagates the result of a simulation through the tree.
        Value is negated at each level because the perspective alternates:
        a good position for the current player is bad for the parent (opponent).
        This is the standard negamax backpropagation used in AlphaGo Zero.'''
        current_value = value_output
        while node is not None:
            node.number_times_chosen += 1
            node.total_v_children += current_value
            node.mean_value_v = node.total_v_children / node.number_times_chosen
            node = node.parent
            current_value = -current_value  # flip perspective for parent

    def run_mcst(self) -> Tuple[int, List[float]]:
        """
        Run the Monte Carlo Search Tree (MCST) algorithm.
        Returns the chosen move index, output policy vector, and training input.

        Dirichlet noise is added to the root's child priors after the first
        iteration populates them, matching AlphaGo Zero's exploration policy:
            p_noisy = (1 - epsilon) * p_network + epsilon * eta
        where eta ~ Dirichlet(alpha). alpha scales with board size so smaller
        boards get proportionally more noise (fewer legal moves).
        """
        _DIRICHLET_EPSILON = 0.25
        # alpha = 0.03 is AGZ's value for 19x19; scale up for smaller boards
        _DIRICHLET_ALPHA   = 0.03 * (19 * 19) / (_BOARD * _BOARD)

        nn_input_backup = self.nn_input_generation(self.root, training=True)
        noise_applied = False

        for idx in range(self.iteration_number):
            selected_node = self.select(self.root, idx)
            value_output = self.expand(selected_node, idx)
            self.backpropagate(selected_node, value_output)

            # inject Dirichlet noise into root child priors on the first iteration
            # that actually produces children — self-play training only
            if not noise_applied and self.training and self.root.children:
                import numpy as np
                n = len(self.root.children)
                noise = np.random.dirichlet([_DIRICHLET_ALPHA] * n)
                for child, eta in zip(self.root.children, noise):
                    child.prior_probability = (
                        (1 - _DIRICHLET_EPSILON) * child.prior_probability
                        + _DIRICHLET_EPSILON * float(eta)
                    )
                noise_applied = True

        output_chances = self.get_choice_info()
        choice_weights = self.get_deep_info()
        the_range = list(range(_MOVES))
        from random import choices
        if self.turn_num < 100:
            the_range = the_range[:-1]
            choice_weights = choice_weights[:-1]
        # suppress first-line moves for the first 100 turns
        # always apply the filter unconditionally — the old conditional check
        # would skip filtering entirely if all non-first-line weights were 0,
        # allowing first-line moves through the all-zeros fallback below.
        if self.turn_num < 100:
            first_line = set()
            for i in range(_BOARD):
                first_line.add(i)                          # row 0
                first_line.add((_BOARD - 1) * _BOARD + i) # row 18
                first_line.add(i * _BOARD)                 # col 0
                first_line.add(i * _BOARD + (_BOARD - 1)) # col 18
            paired = [(v, w) for v, w in zip(the_range, choice_weights)
                      if v not in first_line]
            if paired:
                the_range, choice_weights = zip(*paired)
                the_range      = list(the_range)
                choice_weights = list(choice_weights)
        # if all weights are zero, pick uniformly — use actual move values not indices
        if all(w == 0 for w in choice_weights):
            from random import randrange
            location = [the_range[randrange(len(the_range))]]
        else:
            location = choices(the_range, weights=choice_weights, k=1)
        # root.mean_value_v is the average value from the current player's perspective
        # used by neuralnetboard for resignation decisions
        root_value = self.root.mean_value_v
        return location[0], output_chances, nn_input_backup, root_value

    def get_choice_info(self) -> List[float]:
        '''Returns a list representing the number of times each location was chosen by the MCST.'''
        chance_list: List[float] = [0] * _MOVES
        for spawn in self.root.children:
            if spawn.choice_info[0] == "Pass":
                location = _PASS
            else:
                location = spawn.choice_info[0][0] * _BOARD + spawn.choice_info[0][1]
            chance_list[location] = spawn.number_times_chosen / (self.iteration_number)
        return chance_list

    def get_deep_info(self) -> List[float]:
        '''Returns a list representing the deepmind value for each location on the board.'''
        chance_list: List[float] = [0] * _MOVES
        for spawn in self.root.children:
            spawn_value = self.get_deep_score(spawn)
            if spawn.choice_info[0] == "Pass":
                location = _PASS
            else:
                location = spawn.choice_info[0][0] * _BOARD + spawn.choice_info[0][1]
            chance_list[location] = spawn_value
        return chance_list

    def select(self, node: NNMCSTNode, idx: int) -> NNMCSTNode:
        '''Selects a node for expansion iteratively, generating children when needed.'''
        current = node
        while True:
            if self.is_winning_state(current):
                return current

            if not current.children:
                # leaf node — populate its children then descend into the best one
                self.load_board_string(current)
                legal_moves = self.generate_moves(current)
                _, policy_output = self.child_nn_info(current)
                for move in legal_moves:
                    probability = self.get_probabilities_for_child(policy_output, move)
                    self.generate_child(move, current, idx, probability)
                # if still no children after generation (no legal moves), return this node
                if not current.children:
                    return current
                # loop back to descend into best child
                continue

            return self.select_non_init(current, idx)

    def select_non_init(self, node: NNMCSTNode, idx: int) -> NNMCSTNode:
        '''
        Helper function for the select function, implementing behavior for when the current node is not a winning state,
        or when the root has children.
        '''
        root_child = self.best_child_finder(node)
        while root_child.children:
            root_child = self.best_child_finder(root_child)
        self.load_board_string(root_child)
        if self.is_winning_state(root_child):
            return root_child
        legal_moves = self.generate_moves(root_child)
        _, policy_output = self.child_nn_info(root_child)
        for move in legal_moves:
            probability = self.get_probabilities_for_child(policy_output, move)
            self.generate_child(move, root_child, idx, probability)
        return root_child

    def is_winning_state(self, node: NNMCSTNode) -> bool:
        '''Checks if the current node represents a winning state.'''
        if node.cache_hash[1:] in self.win_cache:
            cache_value = self.win_cache[node.cache_hash[1:]]
            if cache_value[0] == 1:
                return True
            else:
                return False
        else:
            return False

    def expand(self, node: NNMCSTNode, idx) -> float:
        '''Expands the MCST by choosing a move and creating a child node.'''
        self.load_board_string(node)
        if self.is_winning_state(node):
            # return the cached value so backpropagate always receives a float
            cached = self.win_cache.get(node.cache_hash[1:])
            return float(cached[1]) if cached else 0.0
        value_output, policy_output = self.child_nn_info(node)
        legal_move = False
        selected_move = None
        policy_copy = copy.copy(policy_output)
        # track how many non-pass positions we've tried to avoid an infinite loop
        max_tries = _BOARD * _BOARD
        tries = 0
        while not legal_move:
            move = argmax(policy_copy)
            if move == _PASS and self.turn_num < 100:
                # suppress pass during tree search before turn 100 —
                # treat it like an illegal move and try the next best option
                policy_copy[0][move] = -2
                tries += 1
                if tries >= max_tries:
                    selected_move = "Pass"
                    legal_move = True
            elif move != _PASS:
                policy_copy[0][move] = -2
                board_node = self.board[move // _BOARD][move % _BOARD]
                legal_move = self.test_piece_placement(board_node, node)
                selected_move = board_node
                tries += 1
                if tries >= max_tries:
                    # all positions tried and none legal — fall back to pass
                    selected_move = "Pass"
                    legal_move = True
            else:
                selected_move = "Pass"
                legal_move = True
        probability = self.get_probabilities_for_child(policy_output, selected_move)
        self.generate_child(selected_move, node, idx, probability)
        return value_output

    def generate_moves(self, node: NNMCSTNode, simulate=False, final_test=False) -> List[Union[BoardNode, Literal["Pass"]]]:
        '''
        Generates a list of legal moves for the given node based on the current board state.
        Caches the result for future use.
        Returns the possible moves as a List of moves, with moves represented as BoardNodes or as "Pass".
        '''
        if self.cache_hash in self.cache:
            moves = list(self.cache[self.cache_hash])
            if self.turn_num >= 100:
                moves += ["Pass"]
            return moves
        legal_moves: List[Union[BoardNode, Literal["Pass"]]] = []
        if self.turn_num >= 100:
            legal_moves.append("Pass")
        legal_moves_set: Union[Set[None], Set[BoardNode]] = set()
        for board_node in self.board_BoardString.member_set:
            output = self.test_piece_placement(board_node, node, simulate, final_test)
            if output:
                legal_moves.append(board_node)
                legal_moves_set.add(board_node)
        cache_value = frozenset(legal_moves_set)
        self.cache[self.cache_hash] = cache_value

        return legal_moves

    def generate_child(self, selected_move: Union[BoardNode, Literal["Pass"]], node: NNMCSTNode, idx, prob) -> None:
        '''Choose a move and expand the MCST with the selected move.
        When a new node is added, the turn ordering is switched due to a play being already made.'''
        original_board = self.make_board_string()
        if selected_move == "Pass":
            if "Pass" not in node.move_choices.keys():
                node.switch_player()
                # board_list: List[str] format for reload_board_string (board state restoration)
                # history_str: GoBoard str format for ai_training_info_node (NN input)
                board_list = self.make_board_string()
                history_str = self.nn_history_board_string(node)
                temp_train_info = list(node.ai_training_info_node)
                temp_train_info.append(history_str)
                child_node = NNMCSTNode((node.whose_turn, node.not_whose_turn), temp_train_info, prob,
                                        original_board, node.child_killed_last,
                                        ("Pass", idx, node.not_whose_turn.color), parent=node)
                node.children.append(child_node)
                node.move_choices["Pass"] = child_node
                node.switch_player()
            return

        location_tuple = (selected_move.row, selected_move.col)

        if f"{location_tuple}" not in node.move_choices:
            self.expand_play_move(location_tuple, node)
            # board_list: List[str] format for reload_board_string (board state restoration)
            # history_str: GoBoard str format for ai_training_info_node (NN input)
            board_list = self.make_board_string()
            history_str = self.nn_history_board_string(node)
            temp_train_info = list(node.ai_training_info_node)
            temp_train_info.append(history_str)
            child_node = NNMCSTNode((node.whose_turn, node.not_whose_turn), temp_train_info, prob,
                                    board_list, node.child_killed_last,
                                    ((location_tuple[0], location_tuple[1]), idx, node.not_whose_turn.color), parent=node)
            self.reload_board_string(original_board)
            node.children.append(child_node)
            node.move_choices[f"{location_tuple}"] = child_node
            node.switch_player()
        return

    def expand_play_move(self, move, node: NNMCSTNode) -> None:
        '''Expands the MCST by playing the given move on the board.'''
        new_board_piece: BoardNode = self.board[move[0]][move[1]]
        node.child_killed_last.clear()
        self.kill_stones(new_board_piece, node, testing=False)
        new_board_piece.stone_here_color = node.whose_turn.unicode
        node.switch_player()

    def nn_history_board_string(self, node: NNMCSTNode) -> str:
        '''
        Produces a GoBoard-compatible board string for appending to ai_training_info_node.
        Format: single str with turn prefix char + N*N cell chars ('0','1','2').
        This mirrors GoBoard.make_board_string() and is what generate_17_length expects.
        Must NOT use MCST.make_board_string() which returns List[str] (row-per-entry format
        used only for board state restoration via reload_board_string).
        '''
        turn_char = '1' if node.whose_turn.unicode == cf.rgb_black else '2'
        board_str = turn_char
        for row in self.board:
            for cell in row:
                c = cell.stone_here_color
                if c == cf.rgb_black:
                    board_str += '1'
                elif c == cf.rgb_white:
                    board_str += '2'
                else:
                    board_str += '0'
        return board_str

    def child_nn_info(self, node: NNMCSTNode):
        '''Generates the value and policy for a MCSTNode.
        Returns a tuple of a float and an array of length board_size²+1'''
        nn_input = self.nn_input_generation(node)
        val_output, policy_output = neural_net_calcuation(nn_input, _BOARD, self.neural_net_inst)
        return (val_output, policy_output)

    def nn_input_generation(self, node: NNMCSTNode, training: bool = False) -> List[str]:
        '''Generates a list of strings of length 16 or 17 representing the last 8 turns.'''
        nn_input = []
        if node.whose_turn.unicode == cf.rgb_black:
            nn_input = node.ai_training_info_node[-8:]
            nn_input.reverse()
            nn_input.append(self.ai_black_board)
        else:
            nn_input = node.ai_training_info_node[-8:]
            nn_input.reverse()
            nn_input.append(self.ai_white_board)
        if training:
            temp = generate_17_length(nn_input, _BOARD)
            nn_input = temp.tolist()
        return nn_input

    def get_probabilities_for_child(self, policy_output, chosen_bnode: BoardNode) -> float:
        '''Gets the probabilities of choosing a node from the policy_output variable.'''
        if chosen_bnode == "Pass":
            return policy_output[0][_PASS]
        row, col = chosen_bnode.row, chosen_bnode.col
        policy_idx = row * _BOARD + col
        return policy_output[0][policy_idx]
