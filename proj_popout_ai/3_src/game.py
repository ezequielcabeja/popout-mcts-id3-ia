import numpy as np

# Dimensões do tabuleiro
ROWS = 6
COLS = 7


class PopOutGame:
    def __init__(self, rows=ROWS, cols=COLS):        
        self.rows = rows
        self.cols = cols

        # 0: vazio, 1: jogador 1 (X), 2: jogador 2 (O)
        self.board = np.zeros((rows, cols), dtype=int)

        self.current_player = 1 # 1 começa sempre

        # Histórico de estados (para regra de repetição)
        self.history = {}

        # Registar estado inicial
        self._register_state()

    # =========================================================
    # UTILITÁRIOS
    # =========================================================

    def opponent(self, player): # Retorna o oponente do jogador atual
        return 3 - player

    def copy(self): # Cria cópia profunda do jogo (ESSENCIAL para MCTS)        
        new_game = PopOutGame(self.rows, self.cols)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.history = self.history.copy()  # Compartilha histórico para manter contagem de estados
        return new_game

    def _get_state_key(self): # Gera chave única para o estado atual (ESSENCIAL para MCTS)
        return tuple(map(tuple, self.board))

    def _register_state(self): # Registra o estado atual no histórico para contagem de repetições
        state = self._get_state_key()
        self.history[state] = self.history.get(state, 0) + 1

    def is_full(self): # Verifica se o tabuleiro está cheio (ESSENCIAL para regra de empate por tabuleiro cheio)
        return np.all(self.board != 0)

    def display(self): # Exibe o tabuleiro e informações do jogo
        symbols = {0: ".", 1: "X", 2: "O"}

        print("\n" + " ".join(map(str, range(self.cols))))
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))

        print(f"[bold magenta]Vez do jogador: {symbols[self.current_player]}[/bold magenta]")

    # =========================================================
    # MOVIMENTOS
    # =========================================================

    def get_valid_moves(self): # Retorna lista de movimentos válidos (ESSENCIAL para MCTS)
        moves = []

        # Drops
        for c in range(self.cols):
            if self.board[0][c] == 0:
                moves.append(('drop', c))

        # Pops
        for c in range(self.cols):
            if self.board[self.rows - 1][c] == self.current_player:
                moves.append(('pop', c))

        return moves

    def make_move(self, move_type, col): # Executa um movimento
        if (move_type, col) not in self.get_valid_moves():
            return False

        # DROP
        if move_type == 'drop':
            for r in reversed(range(self.rows)):
                if self.board[r][col] == 0:
                    self.board[r][col] = self.current_player
                    break

        # POP
        elif move_type == 'pop':
            for r in range(self.rows - 1, 0, -1):
                self.board[r][col] = self.board[r - 1][col]
            self.board[0][col] = 0

        # Registar novo estado
        self._register_state()

        # Trocar jogador
        self.current_player = self.opponent(self.current_player)

        return True

    def get_next_states(self): # Gera todos os estados possíveis a partir do estado atual (ESSENCIAL para MCTS)
        states = []

        for move in self.get_valid_moves():
            new_game = self.copy()
            new_game.make_move(move[0], move[1])
            states.append((move, new_game))

        return states

    # =========================================================
    # VERIFICAÇÃO DE VITÓRIA
    # =========================================================

    def check_winner(self, move_type=None): # Verifica se há um vencedor, considerando a regra especial do POP

        # Regra especial do POP
        if move_type == 'pop':
            if self._is_winning(self.opponent(self.current_player)):
                return self.opponent(self.current_player)

            if self._is_winning(self.current_player):
                return self.current_player

            return None

        # Verificação normal
        for p in [1, 2]:
            if self._is_winning(p):
                return p

        return None

    def _is_winning(self, player): # Verifica se o jogador tem 4 em linha

        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r][c + i] == player for i in range(4)):
                    return True

        # Vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if all(self.board[r + i][c] == player for i in range(4)):
                    return True

        # Diagonais
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(self.board[r + i][c + i] == player for i in range(4)):
                    return True
                if all(self.board[r + 3 - i][c + i] == player for i in range(4)):
                    return True

        return False

    # =========================================================
    # EMPATES
    # =========================================================

    def check_draw(self): # Verifica condições de empate (repetição de estado e/ ou tabuleiro cheio)
        state = self._get_state_key()

        # Regra 3: repetição
        if self.history.get(state, 0) >= 3:
            return "draw_repetition"

        # Regra 2: tabuleiro cheio
        if self.is_full():
            pop_moves = [c for c in range(self.cols)
                         if self.board[self.rows - 1][c] == self.current_player]

            return "draw_full_board", pop_moves

        return None


# =========================================================
# LOOP DE JOGO (TESTE)
# =========================================================

if __name__ == "__main__": 
    game = PopOutGame()
    game_over = False

    print("[bold bright_blue]Bem-vindo ao PopOut![/bold bright_blue]")

    while not game_over:
        game.display()

        moves = game.get_valid_moves()

        if not moves:
            print("Empate!")
            break

        print(f"Movimentos disponíveis: {moves}")

        try:
            col = int(input("Coluna (0-6): "))
            move_type = input("Tipo (drop/pop): ").strip().lower()

            if not game.make_move(move_type, col):
                print("[bold red]Jogada inválida![/bold red]")
                continue

            winner = game.check_winner( move_type=move_type)

            if winner:
                game.display()
                print(f"[bold green]Jogador {winner} venceu![/bold green]")
                break

            draw_status = game.check_draw()

            if draw_status:
                game.display()

                if draw_status == "draw_repetition":
                    print("[bold red]Empate por repetição![/bold red]")
                    break

                elif draw_status[0] == "draw_full_board":
                    pop_moves = draw_status[1]

                    if pop_moves:
                        print(f"Pops disponíveis: {pop_moves}")
                        choice = input("Digite 'pop' ou 'draw': ").strip()

                        if choice == "draw":
                            print("Empate!")
                            break
                        else:
                            continue
                    else:
                        print("Empate!")
                        break

        except ValueError:
            print("Entrada inválida.")
        except KeyboardInterrupt:
            print("\nJogo interrompido.")
            break

    print("Fim do jogo.")