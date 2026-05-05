import pandas as pd
import numpy as np
import math
from collections import Counter


# =========================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# =========================================================

df = pd.read_csv("../1_data/iris.csv")

# Remover ID (se existir)
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Garantir que atributos são numéricos
for col in df.columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Shuffle + split manual (80/20)
df_shuffle = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df_shuffle) * 0.8)

train_df = df_shuffle.iloc[:split_idx]
test_df = df_shuffle.iloc[split_idx:]


# =========================================================
# 2. ID3 IMPLEMENTAÇÃO
# =========================================================

def entropy(y):
    total = len(y)
    if total == 0:
        return 0

    counts = Counter(y)
    ent = 0

    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)

    return ent


def information_gain(df, attribute, threshold, target="class"):
    total_entropy = entropy(df[target])

    left = df[df[attribute] <= threshold]
    right = df[df[attribute] > threshold]

    if len(left) == 0 or len(right) == 0:
        return 0

    p_left = len(left) / len(df)
    p_right = len(right) / len(df)

    weighted_entropy = (
        p_left * entropy(left[target]) +
        p_right * entropy(right[target])
    )

    return total_entropy - weighted_entropy


def id3_manual(df, attributes, target="class", depth=0, max_depth=4):
    # Caso 1: todos iguais
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # Caso 2: parar
    if depth >= max_depth or not attributes:
        return df[target].mode()[0]

    best_gain = -1
    best_attr = None
    best_threshold = None

    for attr in attributes:
        values = sorted(df[attr].unique())

        thresholds = [
            (values[i] + values[i + 1]) / 2
            for i in range(len(values) - 1)
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
# 3. PREDIÇÃO E VISUALIZAÇÃO
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

        # Melhorar visual
        condition_clean = condition.replace("<=", "≤").replace(">", ">")

        print(f"{indent}{prefix}[{attr} {condition_clean}]")

        new_indent = indent + ("    " if is_last_child else "│   ")

        print_tree(subtree, new_indent, is_last_child)


# =========================================================
# 4. EXECUÇÃO
# =========================================================

print("\nIniciando Treinamento da Árvore ID3...\n")

features = [col for col in df.columns if col != "class"]

tree = id3_manual(train_df, features)

# ----------------------------
# ÁRVORE
# ----------------------------
print("=== ÁRVORE DE DECISÃO (ID3) ===\n")
print_tree(tree)

# ----------------------------
# AVALIAÇÃO
# ----------------------------
corretos = 0

for _, row in test_df.iterrows():
    if predict(tree, row) == row["class"]:
        corretos += 1

accuracy = (corretos / len(test_df)) * 100

print("\n=== RESULTADOS DO MODELO ID3 ===\n")
print(f"{'Total de exemplos:':25} {len(test_df)}")
print(f"{'Corretos:':25} {corretos}")
print(f"{'Incorretos:':25} {len(test_df) - corretos}")

print(f"\nAcurácia no Conjunto de Teste: {accuracy:.2f}%")

# ----------------------------
# EXEMPLO
# ----------------------------
exemplo_teste = {
    "sepallength": 5.1,
    "sepalwidth": 3.5,
    "petallength": 1.4,
    "petalwidth": 0.2
}

resultado = predict(tree, exemplo_teste)

print("\n=== EXEMPLO DE TESTE ===\n")

for k, v in exemplo_teste.items():
    print(f"{k:15}: {v}")

print(f"\n{'Predição:':15} {resultado}")