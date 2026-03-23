
import numpy as np

# Constantes físicas 
mu0  = 4 * np.pi * 1e-7      # H/m
eps0 = 8.854187817e-12        # F/m

# Materiales disponibles 
CONDUCTORES = {
    "1": ("cobre",    5.8e7),
    "2": ("plata",    6.17e7),
    "3": ("aluminio", 3.5e7),
    "4": ("oro",      4.1e7),
    "5": ("otro",     None),
}
DIELECTRICOS = {
    "1": ("polietileno", 2.26, 2e-4),
    "2": ("teflón",      2.1,  3e-4),
    "3": ("aire",        1.0,  0.0),
    "4": ("nylon",       3.0,  2e-2),
    "5": ("otro",        None, None),
}


#  FORMATO DE UNIDADES  

def fmt(valor, ub):
    if valor == 0:
        return f"0 {ub}"
    prefijos = [
        (1e-9,"G"),(1e-6,"M"),(1e-3,"k"),(1e0,""),
        (1e3,"m"),(1e6,"µ"),(1e9,"n"),(1e12,"p"),
    ]
    abs_v = abs(valor)
    for factor, pref in prefijos:
        esc = abs_v * factor
        if 1.0 <= esc < 1000.0:
            mag = int(np.floor(np.log10(esc)))
            dec = max(0, 1 - mag)
            return f"{valor*factor:.{dec}f} {pref}{ub}"
    return f"{valor:.2e} {ub}"

def freq_label(f):
    if f >= 1e9:  return f"{f/1e9:.4g} GHz"
    if f >= 1e6:  return f"{f/1e6:.4g} MHz"
    if f >= 1e3:  return f"{f/1e3:.4g} kHz"
    return f"{f:.4g} Hz"


#  SELECCIÓN AUTOMÁTICA DE RÉGIMEN  BF / AF Bajas Frecuencia O Alta

def f_transicion(a, sigma_c):
    return 1.0 / (np.pi * mu0 * sigma_c * a**2)

def get_regimen(f, ft):
    r = f / ft
    if r <= 1.0:  return "BF"
    if r > 1.0:  return "AF"
    return "TRANS."


def bifilar(a, d, mu, eps, sigma_c, sigma_d, f):
    w     = 2 * np.pi * f
    ln_da = np.log(d / a)           # ln(d/a)
    ft    = f_transicion(a, sigma_c)
    reg   = get_regimen(f, ft)

    if reg == "BF":
        # L = μ/(4π) + (μ/π)·ln(d/a)
        L = mu/(4*np.pi) + (mu/np.pi)*ln_da
        # R = 2 / (σ_c·π·a²)
        R = 2 / (sigma_c * np.pi * a**2)
    else:
        # skin depth: δ = √(2 / (ω·μ0·σ_c))
        delta = np.sqrt(2 / (w * mu0 * sigma_c))
        # L = (μ/π)·ln(d/a)
        L = (mu / np.pi) * ln_da
        # R = 1 / (π·a·δ·σ_c)
        R = 1 / (np.pi * a * delta * sigma_c)

    # C = π·ε / ln(d/a)
    C = (np.pi * eps) / ln_da
    # G = π·σ_d / ln(d/a)
    G = (np.pi * sigma_d) / ln_da

    return L, C, R, G, reg


def coaxial(a, b, c, mu, eps, sigma_c, sigma_d, f):
    w     = 2 * np.pi * f
    ln_ba = np.log(b / a)           # ln(b/a)
    ft    = min(f_transicion(a, sigma_c), f_transicion(b, sigma_c))
    reg   = get_regimen(f, ft)

    if reg == "BF":
        # L = (μ/2π)·{ ln(b/a) + 1/4
        t_ext = (1 / (4*(c**2 - b**2))) * (
                    b**2 - 3*c**2 +
                    (4*c**4 / (c**2 - b**2)) * np.log(c/b)
                )
        L = (mu / (2*np.pi)) * (ln_ba + 0.25 + t_ext)
        # R = (1/σ_c·π)·(1/a² + 1/(c²-b²))
        R = (1 / (sigma_c * np.pi)) * (1/a**2 + 1/(c**2 - b**2))
    else:
        # skin depth: δ = √(2 / (ω·μ0·σ_c))
        delta = np.sqrt(2 / (w * mu0 * sigma_c))
        # L = (μ/2π)·ln(b/a)
        L = (mu / (2*np.pi)) * ln_ba
        # R = 1/(2π·δ·σ_c) · (1/a + 1/b)
        R = (1 / (2*np.pi * delta * sigma_c)) * (1/a + 1/b)

    # C = 2π·ε / ln(b/a)
    C = (2*np.pi*eps) / ln_ba
    # G = 2π·σ_d / ln(b/a)
    G = (2*np.pi*sigma_d) / ln_ba

    return L, C, R, G, reg


#  ENTRADA DE DATOS

def pedir(msg, unidad=""):
    while True:
        try:
            return float(input(f"  {msg}{' ['+unidad+']' if unidad else ''}: ").strip())
        except ValueError:
            print("  Número inválido, intenta de nuevo.")

def menu(titulo, opciones):
    print(f"\n  {titulo}")
    for k, v in opciones.items():
        print(f"    {k}. {v[0]}")
    while True:
        op = input("  Opción: ").strip()
        if op in opciones:
            return op
        print("  Opción no válida.")

def pedir_frecuencia(n):
    print(f"\n  Frecuencia #{n}")
    val = pedir("Valor")
    u   = input("  Unidad (Hz/kHz/MHz/GHz): ").strip().lower()
    return val * {"hz":1,"khz":1e3,"mhz":1e6,"ghz":1e9}.get(u, 1)

#  TABLA DE RESULTADOS

def tabla(filas, frecuencias):
    cw = [max(10, max(len(freq_label(f)) for f in frecuencias)+1),
          13, 13, 13, 13]
    sep = "  +" + "+".join("─"*(w+2) for w in cw) + "+"
    def fila(vals):
        return "  |" + "|".join(f" {str(v):<{w}} " for v,w in zip(vals,cw)) + "|"
    print("\n" + sep)
    print(fila(["Frecuencia","L /m","C /m","R /m","G /m"]))
    print(sep)
    for f,(L,C,R,G,reg) in zip(frecuencias, filas):
        print(fila([freq_label(f), fmt(L,"H"), fmt(C,"F"), fmt(R,"Ω"), fmt(G,"S")]))
    print(sep)


print("\n╔══════════════════════════════════════════╗")
print("║   PARÁMETROS DE LÍNEA DE TRANSMISIÓN    ║")
print("╚══════════════════════════════════════════╝")

tipo = menu("Tipo de línea:", {"1":("Bifilar (two-wire)",),"2":("Coaxial",)})

# Frecuencias
print("\n  ── Frecuencias de operación ──")
n = 0
while not (1 <= n <= 5):
    try: n = int(input("  ¿Cuántas frecuencias? (1-5): ").strip())
    except ValueError: pass
frecuencias = [pedir_frecuencia(i+1) for i in range(n)]

# Dieléctrico
op_d = menu("Dieléctrico:", DIELECTRICOS)
_, eps_r, tan_d = DIELECTRICOS[op_d]
if op_d == "5":
    eps_r = pedir("ε_r")
    tan_d = pedir("tan δ")
else:
    print(f"    → ε_r={eps_r},  tan δ={tan_d}")
eps = eps_r * eps0

# Conductor
op_c = menu("Conductor:", CONDUCTORES)
_, sigma_c = CONDUCTORES[op_c]
if op_c == "5":
    sigma_c = pedir("σ_c","S/m")
else:
    print(f"    → σ_c={sigma_c:.3e} S/m")

# Geometría
if tipo == "1":
    print("\n  ── Geometría bifilar ──")
    a = pedir("Radio del conductor  a","m")
    d = pedir("Separación centros   d","m")
else:
    print("\n  ── Geometría coaxial ──")
    a = pedir("Radio interno          a","m")
    b = pedir("Radio interior externo b","m")
    c = pedir("Radio exterior externo c","m")

# Cálculo por frecuencia
filas = []
for f in frecuencias:
    w      = 2 * np.pi * f
    sigma_d = w * eps * tan_d      # σ_d = ω·ε·tan_δ  (tan_δ cte → σ_d ∝ f)
    mu     = mu0                   # dieléctrico no magnético
    if tipo == "1":
        filas.append(bifilar(a, d, mu, eps, sigma_c, sigma_d, f))
    else:
        filas.append(coaxial(a, b, c, mu, eps, sigma_c, sigma_d, f))

# Resultados
print(f"\n  Resultados — {'Bifilar' if tipo=='1' else 'Coaxial'}")
tabla(filas, frecuencias)
print()