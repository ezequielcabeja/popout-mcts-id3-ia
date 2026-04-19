import pandas as pd
from game import PopOutGame
from mcts import MCTS
import time # IMPORTAÇÃO DE TIME PARA TESTES DE TEMPO (OPCIONAL)
#import os # IMPORTAÇÃO DE OS PARA VER SE O DATASET EXISTE (OPCIONAL)
import random

def board_to_features(board):
    return board.flatten().tolist()

def generate_dataset(n_games=50):
    start = time.time() # INÍCIO DO TIMER (OPCIONAL)

    data = []
    mcts = MCTS(simulations=80) # MCTS COM MENOS SIMULAÇÕES PARA GERAR MAIS RAPIDAMENTE
    mcts_heuristic = MCTS(simulations=40)  # ou outro modo interno
    
    print("\nA gerar dataset...")
    print("...........................\n")

    wins_p1 = 0
    wins_p2 = 0
    draws = 0

    total_moves = 0 # CONTADOR DE JOGADAS PARA CALCULAR MÉDIA DE JOGADAS POR JOGO

    for i in range(n_games):
        if i == 0:
            print(f"Jogo {i+1}/{n_games}")
        elif i == 9:
            print(f"Jogo {i+1}/{n_games}")
        elif (i + 1) % 10 == 0:
            print(f"Jogo {i+1}/{n_games}")

        game = PopOutGame()

        # alternar jogador inicial
        if i % 2 == 1:
           game.current_player = 2

        moves_count = 0
        
        while True:
            features = board_to_features(game.board)

            r = random.random()

            if r < 0.05: # 5% de jogadas aleatórias para aumentar a diversidade do dataset
                move = random.choice(game.get_valid_moves())
            else:
                if moves_count < 10: # USAR HEURÍSTICA NAS PRIMEIRAS JOGADAS PARA GERAR MOVIMENTOS MAIS VARIADOS
                    move = mcts_heuristic.search(game)
                else:
                    move = mcts.search(game) # USAR MCTS NORMAL APÓS AS PRIMEIRAS JOGADAS PARA GERAR MOVIMENTOS MAIS COMPETITIVOS

            move_type, col = move

            label = f"{move_type}_{col}"

            data.append(features + [label])

            game.make_move(move_type, col)
            moves_count += 1
            

            winner = game.check_winner(move_type)
            if winner:
                total_moves += moves_count
                
                if winner == 1:
                    wins_p1 += 1
                else:
                    wins_p2 += 1
                
                break

            draw = game.check_draw()
            if draw:
                total_moves += moves_count # CONTAR A JOGADA QUE LEVOU AO EMPATE
                draws += 1
                break

    columns = [f"f{i}" for i in range(len(features))] + ["label"]
    df = pd.DataFrame(data, columns=columns)

    #os.makedirs("1_data", exist_ok=True)
    df.to_csv("../1_data/dataset.csv", index=False)
    print("\n------------------------------")
    print("Dataset criado!")
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