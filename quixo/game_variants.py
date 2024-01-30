from game import Game, Player
from copy import deepcopy

class GameRL(Game):

    def __init__(self) -> None:
        super().__init__()

    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        trajectory = []
        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            state = deepcopy(self._board)
            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(
                    self)
                ok = self._Game__move(from_pos, slide, self.current_player_idx)
            action = deepcopy((from_pos, slide))
            trajectory.append((state, action))
            winner = self.check_winner()

        return winner, trajectory