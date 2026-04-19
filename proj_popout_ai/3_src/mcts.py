
import random
import math
from game import PopOutGame


class Node: # representa um estado do jogo
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move

        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self): # verifica se todas as jogadas possíveis já foram exploradas
        return len(self.children) == len(self.game.get_valid_moves())

    def best_child(self, c_param=1.4): # calcula o UCT para cada filho e retorna o que tem o maior valor
        choices = []

        for child in self.children:
            uct = (child.wins / (child.visits + 1e-6)) + \
                  c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            choices.append(uct)

        return self.children[choices.index(max(choices))]


class MCTS:
    def __init__(self, simulations=500):
        self.simulations = simulations

    def search(self, root_game):
        root = Node(root_game.copy())

        for _ in range(self.simulations):
            node = root

            # 1. SELECTION
            while node.children and node.is_fully_expanded():
                node = node.best_child()

            # 2. EXPANSION
            if not node.is_fully_expanded():
                moves = node.game.get_valid_moves()

                tried_moves = [child.move for child in node.children]
                for move in moves:
                    if move not in tried_moves:
                        new_game = node.game.copy()
                        new_game.make_move(move[0], move[1])

                        child = Node(new_game, parent=node, move=move)
                        node.children.append(child)
                        node = child
                        break

            # 3. SIMULATION
            result = self.simulate(node.game.copy())

            # 4. BACKPROPAGATION
            while node is not None:
                node.visits += 1

                # vitória do jogador inicial
                if result == root_game.current_player:
                    node.wins += 1

                node = node.parent

        # escolher melhor jogada
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def heuristic_move(self, game):
        moves = game.get_valid_moves()

        # 1. Jogada vencedora imediata
        for move in moves:
            g = game.copy()
            g.make_move(move[0], move[1])
            if g.check_winner() == game.current_player:
                return move

        # 2. Bloquear adversário
        opponent = 3 - game.current_player
        for move in moves:
            g = game.copy()
            g.make_move(move[0], move[1])
            if g.check_winner() == opponent:
                return move

        # 3. Caso contrário → random
        return random.choice(moves)

    def simulate(self, game):
        while True:
            winner = game.check_winner()
            if winner:
                return winner

            draw = game.check_draw()
            if draw:
                return None

            # usar heurística em vez de random puro
            move = self.heuristic_move(game)

            game.make_move(move[0], move[1])
            game.current_player = 3 - game.current_player


