# Seamless Cloning – Poisson Image Editing

Dieses Projekt implementiert Seamless Cloning nach dem Poisson-Verfahren.  
Es besteht aus zwei Hauptteilen:

---

## 📌 1. Laplace-Operator (Finite Differenzen)

In `laplace_operator.py` wird der vektorisierte Laplace-Operator

\[
\Delta = I_M \otimes D_N^{(2)} + D_M^{(2)} \otimes I_N
\]

als dünnbesetzte Sparse-Matrix konstruiert.  
Diese Matrix wird später zur Lösung des Poisson-Problems benötigt.

---

## 📌 2. Seamless Cloning / Poisson Image Editing

In `seamless_cloning.py` werden folgende Verfahren implementiert:

### ✔ Naives Einfügen  
Das zu transferierende Objekt wird direkt in das Zielbild kopiert.

### ✔ Poisson Seamless Cloning  
Das Poisson-Gleichungssystem  
\[
\Delta u = \text{div}(\nabla v)
\]
wird gelöst, um nahtlos Bildbereiche einzufügen.

### ✔ Gemischter Gradient (Mixed Gradients)  
Hier wird für jede Pixelkante der stärkere Gradient aus Quell- und Zielbild übernommen.

---

## 📷 Beispielausgaben

Das Projekt zeigt alle Verfahren anhand zweier Beispielbilder:

- ✈️ Flugzeug  
- 🐻 Bär  

Für jedes Bild werden drei Resultate geplottet:

1. Naives Einfügen  
2. Seamless Cloning (Poisson)  
3. Mixed Gradients  

---

## 🔧 Voraussetzungen

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- skimage

---

## ▶️ Ausführen

