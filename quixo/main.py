import random
from game import Game, Move, Player
from game_variants import GameRL
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm
from copy import deepcopy

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MyPlayer(Player):
    """Player manually controlled by the user"""

    def __init__(self) -> None:
        super().__init__()
        self.find_valid_moves()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        print("YOUR TURN")
        self.print_board(game)

        valid = False

        while not valid:
                
            input_from_pos = input("Insert the position of your move (format: (row, column)): ")
            row, column = map(int, input_from_pos.split(','))
            from_pos = (column, row)

            input_move = input("Insert the slide direction (0: TOP | 1: BOTTOM | 2: LEFT | 3: RIGHT): ")

            if input_move == '0':
                move = Move.TOP
            elif input_move == '1':
                move = Move.BOTTOM                   
            elif input_move == '2':
                move = Move.LEFT
            elif input_move == '3':
                move = Move.RIGHT    
            else:
                move = Move.TOP

            if (from_pos, move) not in self.valid_moves:
                print("Invalid move. Retry.")
            else:
                valid = True

        return from_pos, move
    
    def __name__():
        return 'You'

class MontecarloPlayer(Player):
    """Player that learns how to play according to MonteCarlo learning"""

    def __init__(self, epsilon: float, lr: float, Q: np.array=None) -> None:
        super().__init__()
        self.find_valid_moves()
        if not Q:
            self.Q = defaultdict(float)
        else:
            self.Q = Q
        self.epsilon = epsilon
        self.lr = lr

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_player_idx = game.get_current_player()
        available_moves = [((i, j), slide) for (i, j), slide in self.valid_moves if game._board[j, i] == current_player_idx or game._board[j, i] < 0]
        available_moves_map = [self.map_action(game._board, available_move[0], available_move[1]) for available_move in available_moves]

        # epsilon-greedy strategy
        if np.random.random() < self.epsilon:
            random_move = random.choice(available_moves)
            from_pos = random_move[0]
            move = random_move[1]
        else:
            # generate action a as the best action we can take in state s
            s = self.map_state(game._board, current_player_idx)

            # compute all the possible rotations of a given state and stores only one
            s_array = [s]
            for _ in range(3):
                s_array.append(self.rot_state90(s_array[-1]))
            s = s_array[np.argmin([np.sum(list(s[0])) for s in s_array])]

            # take the best action according to the q-value
            q_star_index = np.argmax([self.Q[(s, a)] for a in available_moves_map])
            # extract from_pos and slide from the best action
            from_pos = available_moves[q_star_index][0]
            move = available_moves[q_star_index][1]
        return from_pos, move
    
    def update_Q(self, game: 'Game', trajectory: list[tuple[tuple[frozenset, frozenset], tuple[int, int]]], reward: float) -> None:
        current_player_idx = game.get_current_player()
        # remove duplicates
        trajectory = list(set([(self.map_state(state, current_player_idx), self.map_action(game._board, action[0], action[1])) for state, action in trajectory]))
        for s, a in trajectory:
            self.Q[(s, a)] += self.lr * (reward - self.Q[(s, a)])

        self.epsilon_decay()

    def epsilon_decay(self):
        self.epsilon = 0.99 * self.epsilon

    def map_state(self, board:np.array, current_player_idx:int) -> tuple[frozenset, frozenset]:
        other_player_idx = 1 - current_player_idx

        # get all the positions where the players have their pieces
        s1 = np.argwhere(board==current_player_idx)
        s2 = np.argwhere(board==other_player_idx)

        # map the positions to an integer in the range [1, 25]
        s1 = map(lambda x: x[0] * board.shape[1] + x[1] + 1, s1)
        s2 = map(lambda x: x[0] * board.shape[1] + x[1] + 1, s2)

        return (frozenset(s1), frozenset(s2))
    
    def map_action(self, board: np.array, from_pos: tuple[int, int], slide: Move) -> tuple[int, int]:
        # map the position to an integer in the range [1, 25]
        from_pos = from_pos[1] * board.shape[1] + from_pos[0] + 1
        return (from_pos, slide)
    
    def rot_state90(self, state: tuple[frozenset, frozenset]) -> tuple[frozenset, frozenset]:
        s1 = (np.array(list(state[0]), dtype=np.uint8) * 5) % 26
        s2 = (np.array(list(state[1]), dtype=np.uint8) * 5) % 26

        return (frozenset(s1), frozenset(s2))
    
    def __name__():
        return 'MonteCarlo'
    
class MinmaxPlayer(Player):

    def __init__(self, max_depth: int, player_idx: int, strategy: str, stopping_criteria: str = 'depth', player_coeffs: list = [1, 20, 100, 1000, 10000],\
                opponent_coeffs: list = [1, 20, 100, 1200, 10000], verbose = False) -> None:
        super().__init__()
        assert strategy in ['min', 'max'], "Strategy must be either min (0) or max (1)"
        assert stopping_criteria in ['depth', 'full'], "The stopping criteria must be either full or depth"
        assert max_depth >= 0, "max_depth must be >= 0"

        self.find_valid_moves()
        self.max_depth = max_depth
        self.strategy = 0 if strategy == 'min' else 1
        self.stopping_criteria = stopping_criteria
        self.player_idx = player_idx
        self.player_coeffs = player_coeffs
        self.opponent_coeffs = opponent_coeffs
        self.verbose = verbose

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        # find all the valid moves
        if self.verbose:
            print("MINMAX TURN")
            self.print_board(game)

        moves = [((i, j), slide) for (i, j), slide in self.valid_moves if game._board[j, i] == self.player_idx or game._board[j, i] < 0]
        
        # evaluate each valid move
        evaluations = []

        for move in moves:
            game_copy = deepcopy(game)
            game_copy._Game__move(move[0], move[1], self.player_idx)
            evaluations.append(self.minmax(game_copy, self.max_depth, 1-self.strategy, 1-self.player_idx))
        
        # find the best move
        best_move_idx = np.argmax(evaluations) if self.strategy == 1 else np.argmin(evaluations)
        best_move = moves[best_move_idx]

        return best_move[0], best_move[1]


    def minmax(self, game: Game, depth: int, strategy: int, current_player_idx: int):
        if depth == 0 or game.check_winner() != -1:
            return self.evaluate_state(game)
        
        # otherwise call recursively minmax for each valid move
        evaluations = []
        moves = [((i, j), slide) for (i, j), slide in self.valid_moves if game._board[j, i] == current_player_idx or game._board[j, i] < 0]

        value = 1 if self.stopping_criteria == 'full' else (1 / (self.max_depth - depth + 1))

        for move in moves:
            if random.random() < value:
                game_copy = deepcopy(game)
                game_copy._Game__move(move[0], move[1], current_player_idx)
                # call minmax decreasing depth, changing strategy and changing current player idx
                evaluations.append(self.minmax(game_copy, depth-1, 1-strategy, 1-current_player_idx))

        if not evaluations:
            return 0

        return max(evaluations) if strategy == 1 else min(evaluations)
        
    def find_sequences(self, board: np.array, current_player_idx):
        values = defaultdict(lambda: 0)

        # sequences in each row
        for r in range(board.shape[0]):
            row = board[r, :]
            row = np.array([1 if val == current_player_idx else 0 for val in row])
            sequences = np.diff(np.where(np.concatenate(([row[0]], row[:-1] != row[1:], [True])))[0])[::2]
            for sequence in sequences:
                values[sequence] += 1

        # sequences in each column
        for c in range(board.shape[1]):
            column = board[:, c]
            column = np.array([1 if val == current_player_idx else 0 for val in column])
            sequences = np.diff(np.where(np.concatenate(([column[0]], column[:-1] != column[1:], [True])))[0])[::2]
            for sequence in sequences:
                values[sequence] += 1

        # main diagonal
        main_diagonal = np.array([1 if board[i, i] == current_player_idx else 0 for i in range(board.shape[0])])
        sequences = np.diff(np.where(np.concatenate(([main_diagonal[0]], main_diagonal[:-1] != main_diagonal[1:], [True])))[0])[::2]
        for sequence in sequences:
            values[sequence] += 1

        anti_diagonal = np.array([1 if board[i, board.shape[0]-i-1] == current_player_idx else 0 for i in range(board.shape[0])])
        sequences = np.diff(np.where(np.concatenate(([anti_diagonal[0]], anti_diagonal[:-1] != anti_diagonal[1:], [True])))[0])[::2]
        for sequence in sequences:
            values[sequence] += 1

        return values
            
    def evaluate_state(self, game: Game):
        player_sequences = self.find_sequences(game._board, self.player_idx)
        opponent_sequences = self.find_sequences(game._board, 1-self.player_idx)

        return self.objective_function(self.player_coeffs, player_sequences) -\
                self.objective_function(self.opponent_coeffs, opponent_sequences)
    
    def objective_function(self, coeffs: list, sequences: list):
        return np.sum([coeffs[i] * sequences[i+1] for i in range(len(coeffs))])
    
    def __name__():
        return 'MinMax'

if __name__ == '__main__':
    
    EPSILON = 1
    LEARNING_RATE = 0.005
    EPISODES = 200_000
    LOAD_Q_TABLE = False
    filename = 'montecarlo_' + str(int(EPISODES/1_000)) + ".pkl"

    if LOAD_Q_TABLE:
        with open(filename, 'rb') as f:
            Q = pickle.load(f)
            player1 = MontecarloPlayer(EPSILON, LEARNING_RATE, Q)
            old_episodes = int(filename.split("_")[1].split(".")[0])
            output_filename = 'montecarlo_' + str(int((EPISODES/1_000)) + old_episodes) + ".pkl"
    else:
        player1 = MontecarloPlayer(EPSILON, LEARNING_RATE)
    player2 = RandomPlayer()

    for i in tqdm(range(EPISODES)):
        g = GameRL()
        winner, trajectory = g.play(player1, player2)
        # define the reward based on the outcome

        if isinstance(player1, MontecarloPlayer):
            reward = 1 if winner == 0 else -1
            # consider only the actions of player 1, i.e., MonteCarlo player. state: board, action: (from_pos, slide)
            trainable_state_action = trajectory[::2]
            # update the Q-table
            player1.update_Q(g, trainable_state_action, reward)
        else:
            reward = 1 if winner == 1 else -1
            # consider only the actions of player 1, i.e., MonteCarlo player. state: board, action: (from_pos, slide)
            trainable_state_action = trajectory[1::2]
            # update the Q-table
            player2.update_Q(g, trainable_state_action, reward)

        # swap the players
        player1, player2 = player2, player1

    # get the Q table from the MontecarloPlayer
    if isinstance(player1, MontecarloPlayer):
        Q = player1.Q
    else:
        Q = player2.Q

    print("Q-table size: ", len(Q))


    # TESTING THE PLAYER
    N_GAMES = 100
    wins = 0

    player1 = MontecarloPlayer(0, 0, Q)
    player2 = RandomPlayer()

    for i in range(N_GAMES):
        g = Game()
        winner = g.play(player1, player2)
        if winner == 0:
            wins += 1

    print("MonteCarlo Player playing first")
    print(f"Win rate: {wins/N_GAMES}%, Losses: {(N_GAMES-wins)/N_GAMES}%")
    print()

    # TESTING THE PLAYER
    N_GAMES = 100
    wins = 0

    player1 = RandomPlayer()
    player2 = MontecarloPlayer(0, 0, Q)

    for i in range(N_GAMES):
        g = Game()
        winner = g.play(player1, player2)
        if winner == 1:
            wins += 1

    print("MonteCarlo Player playing second")
    print(f"Win rate: {wins/N_GAMES}%, Losses: {(N_GAMES-wins)/N_GAMES}%")
    print()

    print("Saving Q-table...")
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)
    print("Q-table successfully saved!")
