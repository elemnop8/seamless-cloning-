""" Modul zum Vergleich von Seamless-Cloning Methoden:
    - Seamless-Cloning mit Laplace-Operator
    - Seamless-Cloning mit gemischten Gradienten
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
from laplace_operator import build_second_derivative_matrix, build_laplace_operator, plot_laplace_operator

def load_image_as_grayscale(filename):
    """
    Lädt ein Bild und konvertiert es zu Graustufen

    Parameter
    ----------
    filename : string
        Pfad zur Bilddatei

    Returns
    -------
    numpy.ndarray
        Graustufenbild als uint8-Array
    """
    img = imread(filename)  # Lädt Bild als numpy array
    if len(img.shape) == 3:  # Wenn RGB Bild
        img_gray = rgb2gray(img)  # Konvertiert zu float [0,1]
        return img_as_ubyte(img_gray)  # Konvertiert zu uint8 [0,255]
    else:
        return img  # Bild ist bereits Graustufen


def build_forward_difference_matrix(n):
    """
    Erstellt die Vorwärtsdifferenzen-Matrix

    Parameter
    ----------
    n : int
        Größe der Matrix

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse-Matrix der Größe n×n
    """
    main = -np.ones(n)
    superd = np.ones(n - 1)
    return sp.diags([main, superd], offsets=[0, 1], shape=(n, n), format='csr')

def build_backward_difference_matrix(n):
    """
    Erstellt die Rückwärtsdifferenzen-Matrix

    Parameter
    ----------
    n : int
        Größe der Matrix

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse-Matrix der Größe n×n
    """
    main = np.ones(n)
    subd = -np.ones(n - 1)
    return sp.diags([main, subd], offsets=[0, -1], shape=(n, n), format='csr')

def build_second_derivative_matrix(n):
    """
    Erstellt die diskrete zweite Ableitungsmatrix

    Parameter
    ----------
    n : int
        Größe der Matrix

    Returns
    -------
    scipy.sparse.csr_matrix
        Tridiagonale Sparse-Matrix der Größe n×n
    """
    data = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
    offsets = [-1, 0, 1]
    return sp.diags(data, offsets, shape=(n, n), format='csr')


def build_laplace_operator(N, M):
    """
    Baut den vektorisierten 2D-Laplace-Operator

    Parameter
    ----------
    N : int
        Anzahl der Zeilen im Bild
    M : int
        Anzahl der Spalten im Bild

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse-Matrix der Größe (N*M × N*M)
    """
    D2_N = build_second_derivative_matrix(N)
    D2_M = build_second_derivative_matrix(M)
    I_N = sp.eye(N, format='csr')
    I_M = sp.eye(M, format='csr')

    term1 = sp.kron(I_M, D2_N, format='csr')
    term2 = sp.kron(D2_M, I_N, format='csr')

    return term1 + term2

def compute_gradient(f):
    """
    Berechnet den diskreten Gradienten eines Bildes

    Parameter
    ----------
    f : numpy.ndarray
        Eingabebild als 2D-Array

    Returns
    -------
    tuple
        Gradient in x- und y-Richtung als zwei 2D-Arrays
    """
    N, M = f.shape
    Dv_N = build_forward_difference_matrix(N)
    Dv_M = build_forward_difference_matrix(M)
    
    grad_x = Dv_N @ f
    grad_y = f @ Dv_M.T

    # Sicherheitshalber Umwandlung in Dense, falls Ergebnis sparse ist
    grad_x = np.asarray(grad_x)
    grad_y = np.asarray(grad_y)

    return grad_x, grad_y

def compute_divergence(v_x, v_y):
    """
    Berechnet die diskrete Divergenz eines Vektorfeldes

    Parameter
    ----------
    v_x : numpy.ndarray
        x-Komponente des Vektorfeldes
    v_y : numpy.ndarray
        y-Komponente des Vektorfeldes

    Returns
    -------
    numpy.ndarray
        Divergenz als 2D-Array
    """
    N, M = v_x.shape
    Dr_N = build_backward_difference_matrix(N)
    Dr_M = build_backward_difference_matrix(M)
    
    term_x = Dr_N @ v_x
    term_y = v_y @ Dr_M.T

    # Sparse in Dense Umwandlung, damit Addition funktioniert
    term_x = np.asarray(term_x)
    term_y = np.asarray(term_y)
    
    div = term_x + term_y
    return div

def compute_gradient_norm(grad_x, grad_y):
    """
    Berechnet die Norm des Gradienten

    Parameter
    ----------
    grad_x : numpy.ndarray
        Gradient in x-Richtung
    grad_y : numpy.ndarray
        Gradient in y-Richtung

    Returns
    -------
    numpy.ndarray
        Norm des Gradienten als 2D-Array
    """
    grad_x = np.asarray(grad_x)
    grad_y = np.asarray(grad_y)

    return np.sqrt(grad_x**2 + grad_y**2)

def seamless_cloning_poisson(f_star, g, position):
    """
    Führt Seamless-Cloning mit Poisson-Gleichung durch

    Parameter
    ----------
    f_star : numpy.ndarray
        Hintergrundbild
    g : numpy.ndarray
        Einzufügendes Bild
    position : tuple
        Position (i0, j0) für die linke obere Ecke

    Returns
    -------
    numpy.ndarray
        Ergebnisbild mit eingefügtem Inhalt
    """
    # Graustufen-Handling falls nötig
    if f_star.ndim == 3:
        f_star = img_as_ubyte(rgb2gray(f_star))
    if g.ndim == 3:
        g = img_as_ubyte(rgb2gray(g))

    i0, j0 = position
    N, M = g.shape
    if i0 + N > f_star.shape[0] or j0 + M > f_star.shape[1]:
        raise ValueError("g passt nicht in f_star an position")

    target_region = f_star[i0:i0+N, j0:j0+M].astype(float)
    g_float = g.astype(float)

    # Laplace-Operator (NM x NM) - entspricht Fortran vektorisierung
    L = build_laplace_operator(N, M)

    # Verwende Fortran-Order beim Vektorisieren (column-major)
    g_vec = g_float.ravel(order='F')
    delta_g_vec = L.dot(g_vec)
    delta_g = delta_g_vec.reshape((N, M), order='F')

    # Inner / boundary mask (innere Punkte sind 1..N-2, 1..M-2)
    inner_mask = np.zeros((N, M), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    boundary_mask = ~inner_mask

    # Indizes in Fortran-Order, damit sie zu L passen
    total_indices = np.arange(N * M).reshape((N, M), order='F')
    inner_indices = total_indices[inner_mask]
    boundary_indices = total_indices[boundary_mask]

    # Submatrizen
    L_II = L[inner_indices, :][:, inner_indices]
    L_IB = L[inner_indices, :][:, boundary_indices]

    boundary_values = target_region.reshape((N*M,), order='F')[boundary_indices]

    rhs = delta_g.reshape((N*M,), order='F')[inner_indices] - L_IB.dot(boundary_values)

    # CG-Lösung
    h_I, info = cg(L_II, rhs, rtol=1e-8, maxiter=2000)
    if info != 0:
        print("CG konvergierte nicht, info:", info)

    # Ergebnis zusammenbauen (reshape in Fortran-Order)
    result_region = target_region.copy().reshape((N*M,), order='F')
    result_region[inner_indices] = h_I
    result_region = result_region.reshape((N, M), order='F')
    result_region = np.round(result_region).clip(0, 255).astype(np.uint8)

    result_image = f_star.copy()
    result_image[i0:i0+N, j0:j0+M] = result_region
    return result_image

def seamless_cloning_mixed_gradients(f_star, g, position):
    """
    Führt Seamless-Cloning mit gemischten Gradienten durch

    Parameter
    ----------
    f_star : numpy.ndarray
        Hintergrundbild
    g : numpy.ndarray
        Einzufügendes Bild
    position : tuple
        Position (i0, j0) für die linke obere Ecke

    Returns
    -------
    numpy.ndarray
        Ergebnisbild mit eingefügtem Inhalt
    """
    # Stelle sicher, dass Bilder Graustufen sind
    if f_star.ndim == 3:
        f_star = img_as_ubyte(rgb2gray(f_star))
    if g.ndim == 3:
        g = img_as_ubyte(rgb2gray(g))

    i0, j0 = position
    N, M = g.shape

    # Überprüfe, ob das Bild in den Zielbereich passt
    if i0 + N > f_star.shape[0] or j0 + M > f_star.shape[1]:
        raise ValueError("g passt nicht in f_star an position")

    # Zielbereich und Quellbild als float
    target_region = f_star[i0:i0+N, j0:j0+M].astype(float)
    g_float = g.astype(float)

    # Gradienten (vorwärts)
    grad_g_x, grad_g_y = compute_gradient(g_float)
    grad_f_x, grad_f_y = compute_gradient(target_region)

    # Wähle den stärkeren Gradienten
    norm_g = np.sqrt(grad_g_x**2 + grad_g_y**2)
    norm_f = np.sqrt(grad_f_x**2 + grad_f_y**2)

    v_x = np.where(norm_f > norm_g, grad_f_x, grad_g_x)
    v_y = np.where(norm_f > norm_g, grad_f_y, grad_g_y)

    # Berechne Divergenz des gemischten Vektorfeldes
    div_v = compute_divergence(v_x, v_y)

    # Aufbau wie bei Poisson: L, Indizes (Fortran)
    L = build_laplace_operator(N, M)

    # Definition der inneren Punkte
    inner_mask = np.zeros((N, M), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    # Indizes in Fortran-Order
    total_indices = np.arange(N * M).reshape((N, M), order='F')
    inner_indices = total_indices[inner_mask]
    boundary_indices = total_indices[~inner_mask]

    # Submatrizen
    L_II = L[inner_indices, :][:, inner_indices]
    L_IB = L[inner_indices, :][:, boundary_indices]

    # Berechne die rechte Seite
    boundary_values = target_region.reshape((N*M,), order='F')[boundary_indices]
    rhs = div_v.reshape((N*M,), order='F')[inner_indices] - L_IB.dot(boundary_values)

    # CG-Lösung
    h_I, info = cg(L_II, rhs, rtol=1e-8, maxiter=2000)
    if info != 0:
        print("CG konvergierte nicht, info:", info)

    # Zusammensetzen des vollständigen Bildausschnitts
    result_region = target_region.copy().reshape((N*M,), order='F')
    result_region[inner_indices] = h_I
    result_region = result_region.reshape((N, M), order='F')
    result_region = np.round(result_region).clip(0, 255).astype(np.uint8)

    # Ergebnisbild zusammenbauen
    result_image = f_star.copy()
    result_image[i0:i0+N, j0:j0+M] = result_region
    return result_image

def main():
    """
    Hauptfunktion zeigt den direkten Vergleich von Seamless Cloning Methoden
    """
    N, M = 5, 7
    plot_laplace_operator(N, M)

    print("Seamless-Cloning - Direkter Vergleich")
    
    # Bilder laden
    print("\nLade Bilder...")
    water = load_image_as_grayscale("water.jpg")
    bear = load_image_as_grayscale("bear.jpg")
    bird = load_image_as_grayscale("bird.jpg")
    plane = load_image_as_grayscale("plane.jpg")
    
    print(f"Bildgrößen: Water{water.shape}, Bear{bear.shape}, Bird{bird.shape}, Plane{plane.shape}")
    
    # Beispiel 1: Bär ins Wasser einfügen
    print("\n--- Bär im Wasser ---")
    position_bear = (40, 30)
    
    # Naives Einfügen
    naive_bear = water.copy()
    H_bear, W_bear = bear.shape
    naive_bear[position_bear[0]:position_bear[0] + H_bear, 
              position_bear[1]:position_bear[1] + W_bear] = bear
    
    # Poisson Cloning
    print("Führe Seamless-Cloning mit Laplace-Operator für Bär durch...")
    result_bear_poisson = seamless_cloning_poisson(water, bear, position_bear)
    
    # Mixed Gradient Cloning
    print("Führe Seamless-Cloning mit gemischten Gradienten für Bär durch...")
    result_bear_mixed = seamless_cloning_mixed_gradients(water, bear, position_bear)
    
    # Beispiel 2: Flugzeug in die Vogellandschaft einfügen
    print("\n--- Flugzeug im Himmel ---")
    
    # Flugzeug zuschneiden
    Ph, Pw = plane.shape
    crop_top  = int(0.15 * Ph)
    crop_bot  = int(0.95 * Ph)
    crop_left = int(0.05 * Pw)
    crop_right= int(0.95 * Pw)
    plane_cropped = plane[crop_top:crop_bot, crop_left:crop_right]
    
    print(f"Flugzeug zugeschnitten: {plane_cropped.shape}")
    
    position_plane = (int(0.15 * bird.shape[0]), int(0.30 * bird.shape[1]))
    
    # Naives Einfügen
    naive_plane = bird.copy()
    H_plane, W_plane = plane_cropped.shape
    naive_plane[position_plane[0]:position_plane[0] + H_plane, 
               position_plane[1]:position_plane[1] + W_plane] = plane_cropped
    
    # Poisson Cloning
    print("Führe Seamless-Cloning mit Laplace-Operator für Flugzeug durch...")
    result_plane_poisson = seamless_cloning_poisson(bird, plane_cropped, position_plane)
    
    # Mixed Gradient Cloning
    print("Führe Seamless-Cloning mit gemischten Gradienten für Flugzeug durch...")
    result_plane_mixed = seamless_cloning_mixed_gradients(bird, plane_cropped, position_plane)
    
    # NUR DIREKTER VERGLEICH - Eine einzige Figur
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Direkter Vergleich: Seamless-Cloning Methoden', fontsize=16)
    
    # Erste Reihe: Bär-Vergleich
    axes[0, 0].imshow(naive_bear, cmap='gray')
    axes[0, 0].set_title('Bär: Naives Einfügen')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(result_bear_poisson, cmap='gray')
    axes[0, 1].set_title('Bär: mit Laplace-Operator ')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(result_bear_mixed, cmap='gray')
    axes[0, 2].set_title('Bär: gemischter Gradient')
    axes[0, 2].axis('off')
    
    # Original Bär
    axes[0, 3].imshow(bear, cmap='gray')
    axes[0, 3].set_title('Original Bär')
    axes[0, 3].axis('off')
    
    # Zweite Reihe: Flugzeug-Vergleich
    axes[1, 0].imshow(naive_plane, cmap='gray')
    axes[1, 0].set_title('Flugzeug: Naives Einfügen')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(result_plane_poisson, cmap='gray')
    axes[1, 1].set_title('Flugzeug: mit Laplace-Operator ')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(result_plane_mixed, cmap='gray')
    axes[1, 2].set_title('Flugzeug: gemischter Gradient')
    axes[1, 2].axis('off')
    
    # Original Flugzeug
    axes[1, 3].imshow(plane_cropped, cmap='gray')
    axes[1, 3].set_title('Original Flugzeug')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    print("\n Der direkte Vergleich zeigt:")
    print("- Von links nach rechts: Naiv -> Laplace-Operator -> gemischter Gradient -> Original")
    plt.savefig(f"Vergleich_SeamlessCloning.png")
    plt.show()


if __name__ == "__main__":
    main()