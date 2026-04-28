import pandas as pd
from game import PopOutGame
from mcts import MCTS
import time # IMPORTAÇÃO DE TIME PARA TESTES DE TEMPO (OPCIONAL)
#import os # IMPORTAÇÃO DE OS PARA VER SE O DATASET EXISTE (OPCIONAL)
import random
from mcts import MCTS, MCTS_Heuristic

def board_to_features(board):
    return board.flatten().tolist()

def generate_dataset(n_games=600):
    start = time.time() # INÍCIO DO TIMER (OPCIONAL)

    data = []
    mcts = MCTS(iterations=60)
    mcts_heuristic = MCTS_Heuristic(iterations=20)  # MCTS HEURÍSTICO PARA AS PRIMEIRAS JOGADAS, POIS SÃO MAIS CRÍTICAS E O MCTS COMPLETO PODE SER MUITO LENTO DEVIDO À GRANDE QUANTIDADE DE POSSIBILIDADES
    
    print("\nA gerar dataset...")
    print("...........................\n")

    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    total_moves = 0 # CONTADOR DE JOGADAS PARA CALCULAR MÉDIA DE JOGADAS POR JOGO

    for i in range(n_games): # GERAR N JOGOS PARA O DATASET
        if i == 0:
            print(f"Jogo {i+1}/{n_games}")
        elif i == 9:
            print(f"Jogo {i+1}/{n_games}")
        elif (i + 1) % 10 == 0:
            print(f"Jogo {i+1}/{n_games}")

        game = PopOutGame()

        # alternar jogador inicial
        if i % 2 == 0:
            game.current_player = 1
        else:
            game.current_player = 2

        moves_count = 0
        
        game_data = [] # PARA ARMAZENAR AS JOGADAS DE CADA JOGO SEPARADAMENTE, SE QUISERES ANALISAR DEPOIS
        while True:
            features = board_to_features(game.board)

            if random.random() < 0.7:
                move = mcts.get_best_move(game)
            else:
                move = mcts_heuristic.get_best_move(game)
            move_type, col = move

            label = f"{move_type}_{col}"

            game_data.append(features + [label])

            game.make_move(move_type, col)
            moves_count += 1
            

            winner = game.check_winner(move_type)
            if winner:
                total_moves += moves_count
                for row in game_data: # ADICIONAR O RESULTADO DE CADA JOGO A CADA LINHA DO DATASET, PARA PODER ANALISAR DEPOIS
                    data.append(row + [winner])
                
                if winner == 1:
                    wins_p1 += 1
                else:
                    wins_p2 += 1
                
                break

            draw = game.check_draw()
            if draw:
                total_moves += moves_count # CONTAR A JOGADA QUE LEVOU AO EMPATE
                draws += 1
                for row in game_data:
                    data.append(row + [0])
                break

    columns = [f"f{i}" for i in range(len(features))] + ["label", "winner"]
    df = pd.DataFrame(data, columns=columns)

    #os.makedirs("1_data", exist_ok=True)
    df.to_csv("../1_data/dataset_popout_game_1.csv", index=False)
    print("\n------------------------------")
    print("dataset_popout_game criado!")
    print("------------------------------\n")

    print(f"Total de jogadas geradas: {len(data)}")

    end = time.time()
    print(f"Tempo total: {end - start:.2f}s")

    avgG = total_moves / n_games
    print(f"Média de jogadas por jogo: {avgG:.2f}")

    avg = (end - start) / n_games
    print(f"Tempo médio por jogo: {avg:.2f}s")

    print("\n------------------------------")
    print(f"\nResultados:")
    print(f"Jogador 1 venceu: {wins_p1}")
    print(f"Jogador 2 venceu: {wins_p2}")
    print(f"Empates: {draws}")
    
if __name__ == "__main__":
    generate_dataset(n_games=50)