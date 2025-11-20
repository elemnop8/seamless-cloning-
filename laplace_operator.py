""" Modul zur Konstruktion des diskreten 2D-Laplace-Operators
    mittels finiter Differenzen.
    Autor: E. Tarielashvili.
    erstellt am: 12.11.2025
    Python 3.12.2
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte

def build_second_derivative_matrix(n):
    """
    Erstellt die diskrete zweite Ableitungsmatrix der Größe n×n

    Parameter
    ----------
    n : int
        Größe der Matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Die tridiagonale Sparse-Matrix der Größe n×n.
    """
    main_diag = -2 * np.ones(n)
    off_diag = np.ones(n-1)
    return sp.diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(n, n), format='csr')

def build_laplace_operator(N, M):
    """
    Baut den vektorisierten 2D-Laplace-Operator

    Parameter
    ----------
    N : int
        Anzahl der Gitterpunkte in x-Richtung.
    M : int
        Anzahl der Gitterpunkte in y-Richtung.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse-Matrix der Größe (N*M×N*M).
    """
    D2_N = build_second_derivative_matrix(N)
    D2_M = build_second_derivative_matrix(M)
    I_N = sp.eye(N, format='csr')
    I_M = sp.eye(M, format='csr')

    # Kronecker-Produkte
    term1 = sp.kron(I_M, D2_N, format='csr')
    term2 = sp.kron(D2_M, I_N, format='csr')

    return term1 + term2

def plot_laplace_operator(N, M, build_laplace_operator=build_laplace_operator, save_path=None, show=True):
    """
    Erstellt und visualisiert den vektorisierten Laplace-Operator für gegebene Dimensionen (N, M).

    Parameter
    ----------
    N : int
        Anzahl der Zeilen.
    M : int
        Anzahl der Spalten.
    build_laplace_operator : callable
        Funktion, die den Laplace-Operator als sparse-Matrix zurückgibt.
    save_path : str, optional
        Dateiname zum Speichern der Grafik (z. B. "laplace_5x7.png"). Wenn None, wird nichts gespeichert.
    show : bool, optional
        Ob die Grafik angezeigt werden soll (Standard: True).
    """
    # Laplace-Operator erstellen
    laplace_operator = build_laplace_operator(N, M)

    # In dichte Matrix umwandeln
    matrix_dense = laplace_operator.toarray()

    # Plot vorbereiten
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis('tight')
    ax.axis('off')

    # Tabelle vorbereiten
    table_data = []
    for i in range(matrix_dense.shape[0]):
        row = []
        for j in range(matrix_dense.shape[1]):
            val = matrix_dense[i, j]
            row.append("" if val == 0 else f"{int(val)}")
        table_data.append(row)

    print(f"Berechne Laplace-Operator für {N}×{M} Bild...")
    print(f"Formel: ∆ = (I_{M} ⊗ D²_{N}) + (D²_{M} ⊗ I_{N})")
    print(f"Matrix-Dimension: {N*M}×{N*M} ({N}×{M} = {N*M} Pixel)")
    print("Visualisiere als Tabelle...")

    # Tabelle zeichnen
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    table.auto_set_column_width([i for i in range(matrix_dense.shape[1])])

    # Titel
    plt.title(f'Vektorisierter Laplace-Operator für (N,M)=({N},{M})', fontsize=14, pad=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Optional speichern
    # plt.savefig(f"Laplace_{N}x{M}.png")
    plt.show()

if __name__ == "__main__":
    N, M = 5, 7
    plot_laplace_operator(N, M)