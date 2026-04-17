from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from mcts import MCTS # IMPORTAÇÃO DO MCTS AQUI
import time # IMPORTAÇÃO DE TIME PARA TESTES DE TEMPO (OPCIONAL)

from utils import GameExit

from game import PopOutGame

console = Console()


class GameInterface:
    def __init__(self):
        self.game = PopOutGame()
        self.mcts = MCTS(simulations=500)

        self.ai_strategies = {
            "MCTS": self.mcts_ai,
            "MCTS_Heuristico": self.mcts_heuristic_ai,
            "Random": self.random_ai
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
        return self.mcts.search(self.game)
    
    def mcts_heuristic_ai(self):
        # podes ter outro objeto ou parâmetro
        return self.mcts.search(self.game)  # já usando heurísticas no mcts.py
    
    def choose_ai(self, player_label):
        console.print(f"\n[bold cyan]Escolha IA {player_label}[/bold cyan]")
        console.print("1 - MCTS")
        console.print("2 - MCTS (Heurístico)")
        console.print("3 - Aleatório")

        choice = Prompt.ask("Opção", choices=["1", "2", "3"])

        mapping = {
            "1": "MCTS",
            "2": "MCTS_Heuristico",
            "3": "Random"
        }

        return mapping[choice]

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
            if ai_type == "MCTS":
                time.sleep(0.8)

            elif ai_type == "MCTS_Heuristico":
                time.sleep(0.5)

            elif ai_type == "Random":
                time.sleep(0.2)

            strategy = self.ai_strategies[ai_type]

            return strategy()
        

    # =========================================================
    # LOOP PRINCIPAL
    # =========================================================
    

    def run(self):
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
                move = self.get_move()

                if move is None:
                    console.print("[red]Erro ao obter jogada[/red]")
                    continue
               # 
                console.print(f"[bold]Jogada escolhida: {move}[/bold]")

                move_type, col = move
                self.game.make_move(move_type, col)

                # -----------------------------
                # RESULTADOS
                # -----------------------------
                winner = self.game.check_winner(last_move_type=move_type)

                if winner:
                    self.render_board()
                    winner_name = self.player_names[winner]
                    console.print(f"[bold green]{winner_name} venceu![/bold green]")
                    
                    break

                draw = self.game.check_draw()

                if draw:
                    self.render_board()
                    console.print("[bold red]Empate[/bold red]")
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
        import random
        return random.choice(self.game.get_valid_moves())


# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":
    interface = GameInterface()
    interface.run()