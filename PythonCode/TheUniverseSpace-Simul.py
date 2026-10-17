# -*- coding: utf-8 -*-
"""
Imagem estática em alta resolução:
Superfície cósmica dos destinos do Universo
Francisco Gonçalves & Augustus Veritas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------- Parâmetros ----------------
H0 = 70.0 / (3.086e19)   # Constante de Hubble em 1/s
Omega_m = 0.3
Omega_x = 0.7

def friedmann(t, a, w):
    adot = H0 * a * np.sqrt(Omega_m * a**(-3) + Omega_x * a**(-3*(1+w)))
    return adot

# Intervalo de tempos
t_max = 5e17
time = np.linspace(0, t_max, 800)

# Valores de w para explorar
w_values = np.linspace(-1.5, -0.2, 25)
results = []

for w in w_values:
    sol = solve_ivp(friedmann, [0, t_max], [1.0], args=(w,), t_eval=time, rtol=1e-8, atol=1e-10)
    t_vals = sol.t / (3.154e16 * 1e9)  # converter para Gyr
    a_vals = sol.y[0]
    mask = a_vals > 0
    results.append((t_vals[mask], a_vals[mask]))

# Preparar arrays alinhados
max_len = max(len(r[0]) for r in results)
T = np.full((len(w_values), max_len), np.nan)
A = np.full((len(w_values), max_len), np.nan)
W = np.full((len(w_values), max_len), np.nan)

for i, (t_vals, a_vals) in enumerate(results):
    T[i, :len(t_vals)] = t_vals
    A[i, :len(a_vals)] = np.clip(a_vals, 0, 6)
    W[i, :len(t_vals)] = w_values[i]

# ---------------- Gráfico 3D ----------------
fig = plt.figure(figsize=(12,9), dpi=200)  # alta resolução
ax = fig.add_subplot(111, projection='3d')

fig.patch.set_facecolor("black")
ax.set_facecolor("black")

surf = ax.plot_surface(T, W, A, cmap="plasma", edgecolor="none", alpha=0.95)

ax.set_xlabel("Tempo (Gyr)", color="white", fontsize=12)
ax.set_ylabel("w (equação de estado)", color="white", fontsize=12)
ax.set_zlabel("Fator de escala a(t)", color="white", fontsize=12)
ax.set_title("Superfície cósmica: destinos possíveis do Universo", color="white", fontsize=14)

ax.tick_params(colors="white")

cbar = fig.colorbar(surf, shrink=0.55, aspect=12)
cbar.set_label("a(t)", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.get_yticklabels(), color="white")

# Melhor ângulo de vista
ax.view_init(elev=30, azim=40)

# Guardar como imagem de alta resolução
plt.savefig("superficie_universo_poster.png", dpi=300, bbox_inches="tight", facecolor="black")

plt.show()
print("✅ Imagem gerada: superficie_universo_poster.png")