
import random
import math
from game import PopOutGame

import pickle
from datetime import datetime


class MCTSNode:
    """Nó da árvore MCTS"""

    def __init__(self, game_state, parent=None, move=None, root_player=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = game_state.get_valid_moves()

        if root_player is not None:
            self.root_player = root_player
        elif parent is not None:
            self.root_player = parent.root_player
        else:
            self.root_player = None

    def uct_value(self, exploration_constant=1.41):#testar valores diferentes!!
        if self.visits == 0:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits + 1) / self.visits
        )

        return exploitation + exploration

    def expand(self):
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        new_game = self.game_state.copy()
        new_game.make_move(move[0], move[1])

        child_node = MCTSNode(new_game, parent=self, move=move, root_player=self.root_player)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_constant=1.20):
        return max(self.children, key=lambda c: c.uct_value(exploration_constant))

    def update(self, result):
        self.visits += 1
        self.wins += result


class MCTS:
    def __init__(self, iterations=700, exploration_constant=1.41):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.dataset = []  # (state, move) para decision tree

    def select(self, node):
        while not node.game_state.check_winner() and node.game_state.get_valid_moves(): #Enquanto o jogo não acabou E ainda existem movimentos possíveis:
            if not node.is_fully_expanded(): #Se o nó atual tem movimentos que ainda não foram explorados:
                return node.expand()
            else:
                node = node.best_child(self.exploration_constant)
        return node

    def simulate(self, game_state, root_player):
        sim_game = game_state.copy()
        
        move_count = 0
        max_moves = 200

        while move_count < max_moves:
            winner = sim_game.check_winner()
            if winner is not None:
                break

            moves = sim_game.get_valid_moves()
            if not moves:
                break

            if move_count < 10:
                drop_moves = [m for m in moves if m[0] == 'drop']
                if drop_moves:
                    moves = drop_moves

            # Ordenar por proximidade ao centro
            moves_sorted = sorted(moves, key=lambda m: abs(m[1] - (sim_game.cols // 2)))
            
            # Escolher uma das 3 melhores jogadas com maior probabilidade
            if random.random() < 0.8 and len(moves_sorted) > 0:
                move = random.choice(moves_sorted[:min(3, len(moves_sorted))])
            else:
                move = random.choice(moves)

            sim_game.make_move(move[0], move[1])
            move_count += 1

        winner = sim_game.check_winner()

        if winner is None:
            return 0.5

        return 1 if winner == root_player else 0

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1 - result  
            node = node.parent

    def get_best_move(self, game_state, return_confidence=False):
        root_player = game_state.current_player   #NOVO
        root = MCTSNode(game_state, root_player=root_player)

        for _ in range(self.iterations):
            node = self.select(root)
            result = self.simulate(node.game_state, root_player)
            self.backpropagate(node, result)

        if not root.children:
            moves = game_state.get_valid_moves()
            if moves:
                best_move = moves[0]
                confidence = 0
            else:
                return None
        else:
            best_child = max(root.children, key=lambda c: c.visits)
            best_move = best_child.move
            confidence = best_child.wins / best_child.visits if best_child.visits > 0 else 0

        # Salvar no dataset
        #self._add_to_dataset(game_state, best_move)

        if return_confidence:
            return best_move, confidence
        return best_move

    def _add_to_dataset(self, game_state, move):
        """Adiciona par (estado, movimento) ao dataset"""
        state_key = {
            'board': game_state.board.tolist(),
            'current_player': game_state.current_player
        }

        self.dataset.append({
            'state': state_key,
            'move': move
        })

    def save_dataset(self, filename=None):
        if filename is None:
            filename = f"mcts_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self.dataset, f)

        print(f"Dataset salvo em {filename} com {len(self.dataset)} exemplos")
        return filename


class MCTS_Heuristic(MCTS):
    """
    MCTS com heurística OTIMIZADA

    """

    def _count_pieces_in_direction(self, board, row, col, player, dr, dc):
        """Conta peças consecutivas em uma direção"""
        count = 0
        r, c = row + dr, col + dc
        rows, cols = len(board), len(board[0])

        while 0 <= r < rows and 0 <= c < cols and board[r][c] == player:
            count += 1
            r += dr
            c += dc

        return count

    def _evaluate_move_fast(self, game_state, move, player):
        """
        Avalia movimento rapidamente (sem cópia!)
        """
        board = game_state.board
        col = move[1]

        # Encontrar linha onde a peça cairia
        row = -1
        for r in range(game_state.rows - 1, -1, -1):
            if board[r][col] == 0:
                row = r
                break

        if row == -1:
            return -1000  # Movimento inválido

        # 1. VITÓRIA IMEDIATA?
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions: 
            count = 1
            count += self._count_pieces_in_direction(board, row, col, player, dr, dc)
            count += self._count_pieces_in_direction(board, row, col, player, -dr, -dc)

            if count >= 4: 
                return 1000  # Vitória!

        # 2. BLOQUEIO? (verificar se oponente venceria)
        opponent = 3 - player
        for dr, dc in directions:
            count = 1
            count += self._count_pieces_in_direction(board, row, col, opponent, dr, dc)
            count += self._count_pieces_in_direction(board, row, col, opponent, -dr, -dc)

            if count >= 3:  # Se oponente tem 3, precisa bloquear
                return 300

        # 3. Avaliar ameaças (3 em linha, 2 em linha)
        score = 0
        for dr, dc in directions:
            count = 1
            count += self._count_pieces_in_direction(board, row, col, player, dr, dc)
            count += self._count_pieces_in_direction(board, row, col, player, -dr, -dc)

            if count == 3:
                score += 100
            elif count == 2:
                score += 10
            elif count == 1:
                score += 1

        # 4. Bônus por posição (centro é melhor)
        if col == 3:
            score += 5
        elif col in [2, 4]:
            score += 3
        elif col in [1, 5]:
            score += 1

        # 5. Bônus por altura (quanto mais baixo, melhor)
        score += (game_state.rows - row) * 2

        return score

    def simulate(self, game_state, root_player):
        sim_game = game_state.copy()
        move_count = 0
        max_moves = 200

        while move_count < max_moves:
            winner = sim_game.check_winner()
            if winner is not None:
                break

            if move_count < 4:
                if random.random() < 0.4:
                    moves = [m for m in sim_game.get_valid_moves() if m[0] == 'drop']
                else:
                    moves = sim_game.get_valid_moves()
            else:
                moves = sim_game.get_valid_moves()
        
            if not moves:
                break

            scored_moves = []
            for move in moves:
                score = self._evaluate_move_fast(sim_game, move, sim_game.current_player)
                scored_moves.append((score, move))

            best_move = max(scored_moves, key=lambda x: x[0])[1]

            if random.random() < 0.6:
                move = best_move
            else:
                move = random.choice(scored_moves[:3])[1]

            sim_game.make_move(move[0], move[1])
            move_count += 1

        winner = sim_game.check_winner()

        if winner is None:
            return 0.5

        return 1 if winner == root_player else 0

class PopOutAI:
    """Wrapper para usar MCTS no jogo"""

    def __init__(self, algorithm='mcts', iterations=500):
        if algorithm == 'mcts':
            self.mcts = MCTS(iterations=iterations)
        elif algorithm == 'mcts_heuristic':
            self.mcts = MCTS_Heuristic(iterations=iterations)
        else:
            self.mcts = MCTS(iterations=iterations)

        self.algorithm_name = algorithm

    def get_move(self, game_state, return_confidence=False):
        return self.mcts.get_best_move(game_state, return_confidence)

    def save_dataset(self, filename=None):
        return self.mcts.save_dataset(filename)

    def get_dataset(self):
        return self.mcts.dataset