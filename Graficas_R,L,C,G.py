"""
=========================================================
 GRAFICAS  R, L, C, G  vs  Frecuencia
 Ejercicio 2-2 : Linea Bifilar  (cobre / polietileno)
 Ejercicio 2-3 : Linea Coaxial  (plata  / teflon)
 Rango: 50 MHz -> 1 GHz
=========================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Constantes ─────────────────────────────────────────
mu0  = 4 * np.pi * 1e-7
eps0 = 8.854187817e-12

# ── Eje de frecuencias: 50 MHz → 1 GHz ────────────────
F_MIN_HZ = 50e6
F_MAX_HZ = 1e9
f     = np.linspace(F_MIN_HZ, F_MAX_HZ, 3000)
w     = 2 * np.pi * f
f_MHz = f / 1e6        # 50 … 1000 MHz

# ══════════════════════════════════════════════════════
#  EJERCICIO 2-2 — LINEA BIFILAR
#  Cobre sc=5.8e7 S/m | a=2mm | d=2cm
#  Polietileno er=2.26 | tan_d=2e-4
# ══════════════════════════════════════════════════════
a_b   = 2e-3
d_b   = 2e-2
sc_b  = 5.8e7
e_b   = 2.26 * eps0
td_b  = 2e-4
ln_da = np.log(d_b / a_b)

# L = (mu/pi)*ln(d/a)   — constante en AF
L_b = (mu0 / np.pi) * ln_da * np.ones_like(f)

# C = pi*eps / ln(d/a)  — constante
C_b = (np.pi * e_b) / ln_da * np.ones_like(f)

# R = 1/(pi*a*delta*sc)    delta = sqrt(2/(w*mu0*sc))
delta_b = np.sqrt(2 / (w * mu0 * sc_b))
R_b     = 1 / (np.pi * a_b * delta_b * sc_b)

# G = pi*sd/ln(d/a)    sd = w*eps*tan_d
G_b = (np.pi * w * e_b * td_b) / ln_da

# ══════════════════════════════════════════════════════
#  EJERCICIO 2-3 — LINEA COAXIAL
#  Plata sc=6.17e7 S/m | a=1.5mm | b=4mm | c=5mm
#  Teflon er=2.1 | tan_d=3e-4
# ══════════════════════════════════════════════════════
a_c   = 1.5e-3
b_c   = 4e-3
sc_c  = 6.17e7
e_c   = 2.1 * eps0
td_c  = 3e-4
ln_ba = np.log(b_c / a_c)

# L = (mu/2pi)*ln(b/a)  — constante en AF
L_c = (mu0 / (2*np.pi)) * ln_ba * np.ones_like(f)

# C = 2pi*eps / ln(b/a) — constante
C_c = (2*np.pi * e_c) / ln_ba * np.ones_like(f)

# R = 1/(2pi*delta*sc) * (1/a + 1/b)
delta_c = np.sqrt(2 / (w * mu0 * sc_c))
R_c     = (1 / (2*np.pi * delta_c * sc_c)) * (1/a_c + 1/b_c)

# G = 2pi*sd/ln(b/a)    sd = w*eps*tan_d
G_c = (2*np.pi * w * e_c * td_c) / ln_ba

# ══════════════════════════════════════════════════════
#  ESTILO
# ══════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#161b27",
    "axes.edgecolor":   "#2e3a52",
    "axes.labelcolor":  "#c9d1e0",
    "axes.titlecolor":  "#e8eaf0",
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "axes.grid":        True,
    "grid.color":       "#1f2c42",
    "grid.linewidth":   0.7,
    "xtick.color":      "#7a8aa0",
    "ytick.color":      "#7a8aa0",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "font.family":      "monospace",
})

COLORES = ["#4fc3f7", "#69f0ae", "#ff7043", "#ce93d8"]

# ══════════════════════════════════════════════════════
#  FUNCION DE GRAFICADO
# ══════════════════════════════════════════════════════
def graficar(f_MHz, f_hz, params, titulo, subtitulo):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(titulo, color="#ffffff", fontsize=13,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.955, subtitulo, ha="center",
             color="#7a8aa0", fontsize=8.5)

    for idx, ax in enumerate(axes.flat):
        nombre, vals, ylabel, escala, unidad = params[idx]
        color = COLORES[idx]
        y = vals * escala

        # curva
        ax.plot(f_MHz, y, color=color, linewidth=2.2)

        # eje X fijado ANTES de cualquier anotacion
        ax.set_xlim(f_MHz[0], f_MHz[-1])

        ax.set_title(nombre, pad=6, color=color, fontweight="bold")
        ax.set_xlabel("Frecuencia  [MHz]", labelpad=4)
        ax.set_ylabel(ylabel)

        # ticks cada 100 MHz
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
        ax.tick_params(which="minor", length=3, color="#2e3a52")

        # líneas verticales de referencia en 100 MHz y 500 MHz
        for fref in [100, 500]:
            ax.axvline(fref, color="#ffffff", linewidth=0.6,
                       linestyle=":", alpha=0.2)

        # puntos + etiqueta en 100 MHz y 1 GHz
        y_range = y.max() - y.min() if y.max() != y.min() else y.max() * 0.1
        for fref_hz in [100e6, 1e9]:
            fref_mhz = fref_hz / 1e6
            i_f = np.argmin(np.abs(f_hz - fref_hz))
            val = vals[i_f] * escala
            ax.plot(fref_mhz, val, "o",
                    color=color, markersize=5, zorder=5, alpha=0.9)
            # etiqueta arriba del punto
            ax.annotate(
                f"{val:.3g} {unidad}",
                xy=(fref_mhz, val),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=7.5, color=color, alpha=0.9,
                ha="center", va="bottom",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

# ── Grafica 1: Bifilar ──────────────────────────────────
params_b = [
    ("R  —  Resistencia",  R_b, "R  [Ohm/m]", 1,    "Ohm/m"),
    ("L  —  Inductancia",  L_b, "L  [nH/m]",  1e9,  "nH/m" ),
    ("C  —  Capacitancia", C_b, "C  [pF/m]",  1e12, "pF/m" ),
    ("G  —  Conductancia", G_b, "G  [nS/m]",  1e9,  "nS/m" ),
]
graficar(f_MHz, f, params_b,
    "Ejercicio 2-2  —  Linea Bifilar",
    "Cobre (sc=5.8e7 S/m)  |  Polietileno (er=2.26, tand=2e-4)  |  a=2mm, d=2cm")

# ── Grafica 2: Coaxial ─────────────────────────────────
params_c = [
    ("R  —  Resistencia",  R_c, "R  [Ohm/m]", 1,    "Ohm/m"),
    ("L  —  Inductancia",  L_c, "L  [nH/m]",  1e9,  "nH/m" ),
    ("C  —  Capacitancia", C_c, "C  [pF/m]",  1e12, "pF/m" ),
    ("G  —  Conductancia", G_c, "G  [uS/m]",  1e6,  "uS/m" ),
]
graficar(f_MHz, f, params_c,
    "Ejercicio 2-3  —  Linea Coaxial",
    "Plata (sc=6.17e7 S/m)  |  Teflon (er=2.1, tand=3e-4)  |  a=1.5mm, b=4mm, c=5mm")

plt.show()