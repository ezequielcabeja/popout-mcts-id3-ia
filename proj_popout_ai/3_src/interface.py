from shutil import move

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from mcts import MCTS # IMPORTAÇÃO DO MCTS AQUI
import time # IMPORTAÇÃO DE TIME PARA TESTES DE TEMPO (OPCIONAL)
from train_id3 import train # IMPORTAÇÃO DO TREINO DE ID3 (SE QUISERES USAR AQUI)
import numpy as np # IMPORTAÇÃO DO NUMPY PARA PREPARAR FEATURES PARA ID3 (SE QUISERES USAR AQUI)
import random
from mcts import MCTS, MCTS_Heuristic

from utils import GameExit

from game import PopOutGame

console = Console()


class GameInterface:
    def __init__(self):
        self.game = PopOutGame()
        self.mcts = MCTS(iterations=500)
        self.mcts_heuristic = MCTS_Heuristic(iterations=1000)

        try: # TENTAR CARREGAR O MODELO ID3 TREINADO (SE EXISTIR)
            self.id3_model = train()
        except:
            self.id3_model = None 

        self.ai_strategies = {
            "MCTS": self.mcts_ai,
            "MCTS_Heuristico": self.mcts_heuristic_ai,
            "Random": self.random_ai,
            "ID3": self.id3_ai
        }

        self.player_ai = {}
        self.player_types = {}

       # self.ai_strategies["NovaIA"] = self.nova_funcao

       
    # =========================================================
    # VISUALIZAÇÃO
    # =========================================================

    def render_board(self):
        table = Table(show_header=True, header_style="bold blue")

        # Cabeçalhos (colunas)
        for col in range(self.game.cols):
            table.add_column(str(col), justify="center")

        symbols = {0: "·", 1: "[red]X[/red]", 2: "[yellow]O[/yellow]"}

        for row in self.game.board:
            table.add_row(*[symbols[cell] for cell in row])

        console.print(table)

        current = self.game.current_player
        name = self.player_names[current]

        console.print(f"\n[bold magenta]Turno: {name} ({symbols[current]})[/bold magenta]")

    # =========================================================
    # INPUT DO UTILIZADOR
    # =========================================================

    def get_player_move(self):
        valid_moves = self.game.get_valid_moves()

        console.print(f"\nMovimentos disponíveis: {valid_moves}")
        console.print("[dim]Digite 'q' ou 'exit' para sair[/dim]")

        while True:
            # NÃO usar "choices" aqui (permite q/exit)
            move_type = Prompt.ask("Tipo de jogada").strip().lower()

            # VERIFICAÇÃO DE SAÍDA (AQUI!)
            if move_type in ["q", "exit"]:
                confirm = Prompt.ask("Tem certeza que quer sair? (y/n)", choices=["y", "n"])
                if confirm == "y":
                    raise GameExit()
                else:
                    continue

            # agora pede coluna
            col = Prompt.ask("Coluna").strip().lower()

            # VERIFICAÇÃO DE SAÍDA (TAMBÉM AQUI!)
            if col in ["q", "exit"]:
                confirm = Prompt.ask("Tem certeza que quer sair? (y/n)", choices=["y", "n"])
                if confirm == "y":
                    raise GameExit()
                else:
                    continue

            # valida número
            try:
                col = int(col)
            except:
                console.print("[red]Coluna inválida[/red]")
                continue

            # valida tipo de jogada
            if move_type not in ["drop", "pop"]:
                console.print("[red]Tipo inválido (use drop/pop)[/red]")
                continue

            # valida jogada completa
            if (move_type, col) in valid_moves:
                return move_type, col
            else:
                console.print("[red]Jogada inválida. Tente novamente.[/red]")

    # =========================================================
    # MODOS DE JOGO
    # =========================================================
    def mcts_ai(self):
        return self.mcts.get_best_move(self.game)

    def mcts_heuristic_ai(self):
        return self.mcts_heuristic.get_best_move(self.game)
    
    def choose_ai(self, player_label):
        console.print(f"\n[bold cyan]Escolha IA {player_label}[/bold cyan]")
        console.print("1 - MCTS")
        console.print("2 - MCTS (Heurístico)")
        console.print("3 - Aleatório")
        console.print("4 - ID3 (Aprendido)")

        choice = Prompt.ask("Opção", choices=["1", "2", "3", "4"])

        mapping = {
            "1": "MCTS",
            "2": "MCTS_Heuristico",
            "3": "Random",
            "4": "ID3"
            }

        return mapping[choice]
    
    def id3_ai(self):

        if self.id3_model is None:
            console.print("[red]ID3 não disponível → fallback[/red]")
            return self.random_ai()
        
        features = self.game.board.flatten().reshape(1, -1)
        prediction = self.id3_model.predict(features)[0]

        move_type, col = prediction.split("_")
        col = int(col)

        if (move_type, col) in self.game.get_valid_moves():
            return (move_type, col)
        else:
            return random.choice(self.game.get_valid_moves())

    def choose_mode(self):
        console.print("\n[bold cyan]Escolha o modo de jogo:[/bold cyan]")
        console.print("1 - Humano_A vs Humano_B")
        console.print("2 - Humano_A vs IA")
        console.print("3 - IA vs IA")

        choice = Prompt.ask("Opção", choices=["1", "2", "3"])
        return int(choice)
        
    def get_move(self):
        player = self.game.current_player

        if self.player_types[player] == "human":
            return self.get_player_move()

        elif self.player_types[player] == "ai":
            ai_type = self.player_ai[player]

            #TESTE DE TEMPO (OPCIONAL)
            time.sleep(0.5) # pequena pausa para melhor experiência visual

            strategy = self.ai_strategies[ai_type]

            return strategy()
        

    # =========================================================
    # LOOP PRINCIPAL
    # =========================================================
    

    def run(self):
        moves_count = 0 # CONTADOR DE JOGADAS PARA TESTES DE TEMPO (OPCIONAL)
        try:
            mode = self.choose_mode()
            
            # Reset jogo
            self.game = PopOutGame()

            if mode == 1:
                self.player_types = {1: "human", 2: "human"}
                self.player_names = {1: "Humano_A", 2: "Humano_B"}

            elif mode == 2:
                ai_type = self.choose_ai("Jogador 2")

                self.player_types = {1: "human", 2: "ai"}
                self.player_names = {1: "Humano_A", 2: ai_type}
                self.player_ai = {2: ai_type}

            elif mode == 3:
                ai1 = self.choose_ai("Jogador 1")
                ai2 = self.choose_ai("Jogador 2")

                self.player_types = {1: "ai", 2: "ai"}
                self.player_names = {1: ai1, 2: ai2}
                self.player_ai = {1: ai1, 2: ai2}

            while True:
                self.render_board()

                current_name = self.player_names[self.game.current_player]
                console.print(f"[dim]{current_name} está a jogar...[/dim]")

                # -----------------------------
                # ESCOLHER JOGADA
                # -----------------------------
                move = self.get_move()   # PRIMEIRO obter jogada

                # depois validar
                if move is None:
                    console.print("[red]Erro ao obter jogada[/red]")
                    continue

                valid_moves = self.game.get_valid_moves()

                if move not in valid_moves:
                    console.print("[yellow]Jogada inválida → fallback[/yellow]")
                    move = random.choice(valid_moves)
               # 
                console.print(f"[bold]Jogada escolhida: {move}[/bold]")

                move_type, col = move
                self.game.make_move(move_type, col)
                moves_count += 1 # CONTADOR DE JOGADAS (OPCIONAL)

                # -----------------------------
                # RESULTADOS
                # -----------------------------
                winner = self.game.check_winner(move_type=move_type) 

                if winner:
                    self.render_board()
                    winner_name = self.player_names[winner]
                    symbol = "X" if winner == 1 else "O"
                    console.print(f"[bold green]{winner_name} ({symbol}) venceu! 🏆[/bold green]")
                    console.print(f"[cyan]Total de jogadas: {moves_count}[/cyan]")
                    
                    break

                draw = self.game.check_draw()

                if draw:
                    self.render_board()
                    console.print("[bold red]Empate[/bold red]")
                    console.print(f"[cyan]Total de jogadas: {moves_count}[/cyan]")
                    break

        except GameExit:
            console.print("\n[bold yellow]Jogo terminado pelo utilizador.[/bold yellow]")

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrompido (Ctrl+C).[/bold yellow]")

        finally:
            console.print("[dim]Obrigado por jogar PopOut![/dim]")

    # =========================================================
    # Random SIMPLES (TEMPORÁRIA)
    # =========================================================

    def random_ai(self):
        return random.choice(self.game.get_valid_moves())


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":
    interface = GameInterface()
    interface.run()