# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def edit_distance(x, y):
    m, n = len(x), len(y)
    D = np.zeros((m+1, n+1), dtype=np.int32)

    for i in range(m+1):
        D[i, 0] = i
    for j in range(n+1):
        D[0, j] = j

    for i in range(m):
        for j in range(n):
            cost = 0 if x[i] == y[j] else 1
            D[i+1, j+1] = min(
                D[i,   j+1] + 1,   # deletion
                D[i+1, j] + 1,   # insertion
                D[i,   j] + cost  # substitution
            )
    return int(D[m, n])


def confusion_matrix(true, pred, ignore=[]):
    # Get all possible class labels, ignoring those in `ignore`
    classes = sorted(set(true) | set(pred))
    classes = [c for c in classes if c not in ignore]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    mat = np.zeros((n, n), dtype=int)

    for t, p in zip(true, pred):
        if t in ignore or p in ignore:
            continue
        i = class_to_idx[t]
        j = class_to_idx[p]
        mat[i, j] += 1

    return mat, classes


def plot_confusion_matrix(mat, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prediction')
    plt.ylabel('Verite')
    plt.title('Matrice de Confusion')
    plt.show()
