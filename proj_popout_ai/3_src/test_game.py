from rich.console import Console
from rich.table import Table
import time
import os

from game import PopOutGame
from mcts import MCTS, MCTS_Heuristic
from id3_popout_dataset import load_model, predict

import random


console = Console()


# =========================================================
# ID3 AI
# =========================================================

class ID3_AI:
    def __init__(self):
        self.model = load_model()

    def board_to_dict(self, game):
        sample = {}

        rows, cols = game.board.shape

        for r in range(rows):
            for c in range(cols):
                sample[f"cell_{r}_{c}"] = game.board[r][c]

        sample["current_player"] = game.current_player

        return sample

    def get_move(self, game):

        sample = self.board_to_dict(game)

        prediction = predict(self.model, sample)

        move_type, col = prediction.split("_")
        col = int(col)

        move = (move_type, col)

        if move in game.get_valid_moves():
            return move

        return random.choice(game.get_valid_moves())


# =========================================================
# RANDOM AI
# =========================================================

class RandomAI:
    def get_move(self, game):
        return random.choice(game.get_valid_moves())


# =========================================================
# MCTS AI
# =========================================================

class MCTS_AI:
    def __init__(self, iterations=700):
        self.mcts = MCTS(iterations=iterations)

    def get_move(self, game):
        return self.mcts.get_best_move(game)


# =========================================================
# MCTS HEURISTIC AI
# =========================================================

class MCTS_Heuristic_AI:
    def __init__(self, iterations=400):
        self.mcts = MCTS_Heuristic(iterations=iterations)

    def get_move(self, game):
        return self.mcts.get_best_move(game)


# =========================================================
# GAME SIMULATION
# =========================================================

def play_game(ai1, ai2):

    game = PopOutGame()

    total_moves = 0

    while True:

        current_player = game.current_player

        if current_player == 1:
            move = ai1.get_move(game)
        else:
            move = ai2.get_move(game)

        # fallback
        if move not in game.get_valid_moves():
            move = random.choice(game.get_valid_moves())

        move_type, col = move

        game.make_move(move_type, col)

        total_moves += 1

        winner = game.check_winner(move_type=move_type)

        if winner:
            return winner, total_moves

        if game.check_draw():
            return 0, total_moves


# =========================================================
# MATCH SERIES
# =========================================================

def evaluate(ai_name, ai, games=50):

    console.print(f"\n[bold cyan]ID3 vs {ai_name}[/bold cyan]")

    id3 = ID3_AI()

    id3_wins = 0
    opponent_wins = 0
    draws = 0

    total_moves = 0

    start_time = time.time()

    for i in range(games):

        # alternar quem começa
        if i % 2 == 0:
            winner, moves = play_game(id3, ai)

            if winner == 1:
                id3_wins += 1
            elif winner == 2:
                opponent_wins += 1
            else:
                draws += 1

        else:
            winner, moves = play_game(ai, id3)

            if winner == 1:
                opponent_wins += 1
            elif winner == 2:
                id3_wins += 1
            else:
                draws += 1

        total_moves += moves

    total_time = time.time() - start_time

    # =====================================================
    # GUARDAR RELATÓRIO
    # =====================================================

    os.makedirs("../4_reports", exist_ok=True)

    report_path = f"../4_reports/id3_vs_{ai_name}.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write(f"RESULTADOS: ID3 vs {ai_name}\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Jogos: {games}\n")
        f.write(f"Vitórias ID3: {id3_wins}\n")
        f.write(f"Vitórias {ai_name}: {opponent_wins}\n")
        f.write(f"Empates: {draws}\n")

        winrate = (id3_wins / games) * 100

        f.write(f"Win Rate ID3: {winrate:.2f}%\n")
        f.write(f"Média de Jogadas: {total_moves / games:.2f}\n")
        f.write(f"Tempo Médio/Jogo: {total_time / games:.2f}s\n")

    # =====================================================
    # TABELA
    # =====================================================

    table = Table(title=f"RESULTADOS: ID3 vs {ai_name}")

    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Jogos", str(games))
    table.add_row("Vitórias ID3", str(id3_wins))
    table.add_row(f"Vitórias {ai_name}", str(opponent_wins))
    table.add_row("Empates", str(draws))

    winrate = (id3_wins / games) * 100

    table.add_row("Win Rate ID3", f"{winrate:.2f}%")

    table.add_row(
        "Média de Jogadas",
        f"{total_moves / games:.2f}"
    )

    table.add_row(
        "Tempo Médio/Jogo",
        f"{total_time / games:.2f}s"
    )

    console.print(table)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate(
        "Random",
        RandomAI(),
        games=50
    )

    evaluate(
        "MCTS",
        MCTS_AI(iterations=2000),
        games=50
    )

    evaluate(
        "MCTS_Heuristico",
        MCTS_Heuristic_AI(iterations=700),
        games=50
    )