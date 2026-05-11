import pandas as pd
import numpy as np
import math
from collections import Counter

import pickle


# =========================================================
# 1. CARREGAMENTO
# =========================================================

df = pd.read_csv("../1_data/dataset_popout_game_2.csv")

# separar treino/teste
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]


# =========================================================
# 2. ID3
# =========================================================

def entropy(y):
    total = len(y)
    counts = Counter(y)

    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)

    return ent


def information_gain(df, attribute, threshold, target="label"):
    total_entropy = entropy(df[target])

    left = df[df[attribute] <= threshold]
    right = df[df[attribute] > threshold]

    if len(left) == 0 or len(right) == 0:
        return 0

    p_left = len(left) / len(df)
    p_right = len(right) / len(df)

    return total_entropy - (
        p_left * entropy(left[target]) +
        p_right * entropy(right[target])
    )


def id3_manual(df, attributes, target="label", depth=0, max_depth=3):

    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    if depth >= max_depth or not attributes:
        return df[target].mode()[0]

    best_gain = -1
    best_attr = None
    best_threshold = None

    for attr in attributes:
        values = sorted(df[attr].unique())

        thresholds = [
            (values[i] + values[i+1]) / 2
            for i in range(len(values)-1)
        ]

        for t in thresholds:
            gain = information_gain(df, attr, t, target)

            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_threshold = t

    if best_gain <= 0:
        return df[target].mode()[0]

    node_key = (best_attr, best_threshold)
    tree = {node_key: {}}

    left_df = df[df[best_attr] <= best_threshold]
    right_df = df[df[best_attr] > best_threshold]

    tree[node_key][f"<= {best_threshold:.2f}"] = id3_manual(
        left_df, attributes, target, depth + 1
    )

    tree[node_key][f"> {best_threshold:.2f}"] = id3_manual(
        right_df, attributes, target, depth + 1
    )

    return tree


# =========================================================
# 3. PREDIÇÃO
# =========================================================

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    (attr, threshold), branches = next(iter(tree.items()))

    val = sample[attr]

    if val <= threshold:
        return predict(branches[f"<= {threshold:.2f}"], sample)
    else:
        return predict(branches[f"> {threshold:.2f}"], sample)


def board_to_dict(board):
    flat = board.flatten()
    return {f"f{i}": flat[i] for i in range(len(flat))}


# =========================================================
# 4. VISUALIZAÇÃO
# =========================================================

def print_tree(tree, indent="", is_last=True):
    if not isinstance(tree, dict):
        prefix = "└── " if is_last else "├── "
        print(f"{indent}{prefix}CLASSE: {tree}")
        return

    (attr, threshold), branches = next(iter(tree.items()))
    items = list(branches.items())

    for i, (condition, subtree) in enumerate(items):
        is_last_child = (i == len(items) - 1)
        prefix = "└── " if is_last_child else "├── "

        condition_clean = condition.replace("<=", "≤")

        print(f"{indent}{prefix}[{attr} {condition_clean}]")

        new_indent = indent + ("    " if is_last_child else "│   ")

        print_tree(subtree, new_indent, is_last_child)


def train():
    df = pd.read_csv("../1_data/dataset_popout_game_2.csv")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]

    features = [
                col for col in df.columns
                if col not in ["label", "winner"]
            ]

    tree = id3_manual(train_df, features)

    return tree



def save_model(tree, path="id3_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(tree, f)

def load_model(path="id3_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
if __name__ == "__main__":

    print("\nIniciando Treinamento ID3 (PopOut)...\n")

    df = pd.read_csv("../1_data/dataset_popout_game_2.csv")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    features = [
                col for col in df.columns
                if col not in ["label", "winner"]
            ]

    tree = id3_manual(train_df, features)

    print("=== ÁRVORE DE DECISÃO (PopOut) ===\n")
    print_tree(tree)

    # =============================
    # AVALIAÇÃO
    # =============================
    corretos = 0

    for _, row in test_df.iterrows():
        sample = row.to_dict()
        if predict(tree, sample) == row["label"]:
            corretos += 1

    accuracy = (corretos / len(test_df)) * 100

    print("\n=== RESULTADOS DO MODELO ID3 ===\n")
    print(f"{'Total de exemplos:':25} {len(test_df)}")
    print(f"{'Corretos:':25} {corretos}")
    print(f"{'Incorretos:':25} {len(test_df) - corretos}")

    print(f"\nAcurácia no Conjunto de Teste: {accuracy:.2f}%")

    # =============================
    # TESTE
    # =============================
    print("\n=== TESTE COM TABULEIRO REAL ===\n")

    example_board = np.zeros((6, 7))
    sample = board_to_dict(example_board)

    prediction = predict(tree, sample)

    print(f"{'Predição:':15} {prediction}")

    # =============================
    # GUARDAR MODELO (MUITO IMPORTANTE)
    # =============================
    save_model(tree)
    print("\n[✔] Modelo guardado como id3_model.pkl")

    