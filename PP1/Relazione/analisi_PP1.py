#!/usr/bin/env python3
"""Analisi dei dati della prova pratica PP1: calore specifico e calore latente.

Versione del 14/10/2026: figura del fit di M_e separata (parte facoltativa), numeri per gli
esempi di calcolo e per le spiegazioni della relazione. 18/10/2026: numeri per le risposte ai
commenti del gruppo (test sulla prova anomala, contributi all'incertezza, stime spiegate passo
per passo). 18/10/2026, sera: T_f del calore latente dal quaderno del gruppo (pagina 3).

Legge i dati in dati/*.csv, copiati dal logbook con uno script (il logbook non si
modifica), e scrive:
  generati/numeri.tex   macro con i risultati citati nel testo della relazione
  generati/tab_*.tex    corpi delle tabelle (righe tra \\midrule e \\bottomrule)
  figure/*.pdf          grafici
Si lancia dalla cartella Relazione:  python3 analisi_PP1.py

Convenzioni: calori specifici in cal/(g K), calore latente in cal/g, masse in g,
temperature in gradi Celsius (le differenze di temperatura sono uguali in K).
Incertezze: deviazioni standard; tipo B da risoluzione = RIS/sqrt(12); propagazione
al primo ordine con le covarianze (statistica di Laboratorio di meccanica).
"""
import csv
from math import floor, log10, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2 as chi2dist, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QUI = Path(__file__).resolve().parent
DATI, GEN, FIG = QUI / "dati", QUI / "generati", QUI / "figure"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Costanti e valori di riferimento
# --------------------------------------------------------------------------
CA = 1.0                 # cal/(g K), calore specifico dell'acqua (scheda)
J_CAL = 4.186            # J per caloria (Focardi p. 509)
ME, S_ME = 25.0, 5.0     # g, massa equivalente dei thermos assunta (scheda)
RIS_M, RIS_T = 0.1, 0.2  # risoluzioni di bilancia (g) e termometri (gradi C)
SB_M, SB_T = RIS_M / sqrt(12), RIS_T / sqrt(12)
# T_f del calore latente: nel logbook arrotondate al grado (24, 21, 30, 19); sul quaderno del
# gruppo (pagina 3, foto del 18/10/2026) lette al decimo, interpolando tra due tacche. Si usano i
# valori del quaderno; il CSV estratto dal logbook resta com'e'. Con i valori del logbook, trattati
# come arrotondati al grado (1 grado/sqrt(12) in quadratura con sigma_T), si calcola solo il
# confronto della tabella delle differenze (lamLogbook).
TF_QUADERNO = {1: 24.1, 2: 21.1, 3: 29.9, 4: 18.9}
SB_TF_LOG = 1.0 / sqrt(12)

# Al: Focardi tab. 13-2; ottone, granito, quarzo: Engineering ToolBox (J/(g K) -> cal/(g K))
C_REF = {"Al": 0.215, "Ott": round(0.375 / J_CAL, 3), "granito": round(0.790 / J_CAL, 2),
         "quarzo": round(0.730 / J_CAL, 3)}
# rocce comuni (arenaria, ardesia, granito, marmo, calcare): da 0.710 a 0.909 J/(g K)
ROCCE = (0.710 / J_CAL, 0.909 / J_CAL)
LAMBDA_REF = 79.7        # cal/g (Focardi p. 522, Mencuccini p. 435)

NOMI = {"Al": "Alluminio", "Ott": "Metallo ignoto", "RB": "Roccia bianca", "RG": "Roccia grigia"}
ESCLUSE_C = {("Al", 1)}  # prova anomala del calore specifico (vedi relazione)
ESCLUSE_L = {1}          # prova anomala del calore latente


def leggi(nome):
    with open(DATI / nome, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


thermos = {r["thermos"]: float(r["massa_vuoto_g"]) for r in leggi("thermos.csv")}
masse = {}
for r in leggi("masse_campioni.csv"):
    p = np.array([float(r[k]) for k in ("pesata1_g", "pesata2_g", "pesata3_g")])
    # media delle tre pesate; incertezza: tipo A della media e tipo B della bilancia
    sa = p.std(ddof=1) / sqrt(len(p))
    masse[r["campione"]] = (p.mean(), sqrt(sa**2 + SB_M**2), p)


# --------------------------------------------------------------------------
# Strumenti statistici
# --------------------------------------------------------------------------
def media_pesata(v, s):
    v, s = np.asarray(v, float), np.asarray(s, float)
    w = 1 / s**2
    m = np.sum(w * v) / np.sum(w)
    chi2 = float(np.sum(w * (v - m) ** 2))
    return m, 1 / sqrt(np.sum(w)), chi2, len(v) - 1, w / np.sum(w)


def fmt_p(p):
    """Probabilita' in percento, con due cifre significative."""
    q = 100 * p
    if q >= 10:
        return f"{q:.0f}"
    return f"{q:.2g}" if q >= 0.001 else "<0.001"


def p_chi2(chi2, ndf):
    """Probabilita' di ottenere un chi quadro maggiore di quello osservato."""
    return float(chi2dist.sf(chi2, ndf)) if ndf > 0 else float("nan")


def fit_lineare(x, y, s):
    """Fit pesato y = a + b x (formule di Laboratorio di meccanica)."""
    x, y, s = map(lambda t: np.asarray(t, float), (x, y, s))
    p = 1 / s**2
    P = p.sum()
    xm, ym = (p * x).sum() / P, (p * y).sum() / P
    x2m, xym = (p * x * x).sum() / P, (p * x * y).sum() / P
    varx = x2m - xm**2
    b = (xym - xm * ym) / varx
    a = ym - b * xm
    vb = 1 / (P * varx)
    va = x2m * vb
    cov = -xm * vb
    chi2 = float((p * (y - a - b * x) ** 2).sum())
    return a, b, sqrt(va), sqrt(vb), cov, chi2, len(x) - 2


# --------------------------------------------------------------------------
# Calore specifico, eq. (3) della scheda
# --------------------------------------------------------------------------
cs = []
for r in leggi("calore_specifico.csv"):
    k, n = r["campione"], int(r["prova"])
    m1, s_m1, _ = masse[k]
    ma = float(r["thermos_acqua_g"]) - thermos[r["thermos"]]
    T1, Ta, Tf = (float(r[c]) for c in ("T1_C", "Ta_C", "Tf_C"))
    cs.append(dict(k=k, n=n, W=float(r["thermos_acqua_g"]), m1=m1, s_m1=s_m1, ma=ma,
                   s_ma=sqrt(2) * SB_M, T1=T1, Ta=Ta, Tf=Tf, dT1=Tf - Ta, dT2=T1 - Tf,
                   escl=(k, n) in ESCLUSE_C))


def c_di(d, me=ME):
    return CA * (d["ma"] + me) * d["dT1"] / (d["m1"] * d["dT2"])


def sigma_c(d, sT):
    """Incertezza statistica di c_1 per una prova (M_e a parte, sistematica comune).
    Proposizione 3.12 degli appunti: T_f compare in entrambe le differenze."""
    nu2 = ((d["s_ma"] / (d["ma"] + ME)) ** 2 + (d["s_m1"] / d["m1"]) ** 2
           + (sT / d["dT1"]) ** 2 + (sT / d["dT2"]) ** 2
           + sT**2 * (1 / d["dT1"] + 1 / d["dT2"]) ** 2)
    return c_di(d) * sqrt(nu2)


for d in cs:
    d["c"] = c_di(d)
    d["dc_dMe"] = d["c"] / (d["ma"] + ME)
    d["s_c_ris"] = sigma_c(d, SB_T)

al = [d for d in cs if d["k"] == "Al" and not d["escl"]]
chi2_al_ris = media_pesata([d["c"] for d in al], [d["s_c_ris"] for d in al])[2]


def chi2_rid_al(sT):
    _, _, chi2, ndf, _ = media_pesata([d["c"] for d in al], [sigma_c(d, sT) for d in al])
    return chi2 / ndf - 1


# incertezza efficace di una lettura di temperatura: quella che rende la dispersione
# delle prove sull'alluminio compatibile con le incertezze (chi quadro ridotto = 1)
ST = brentq(chi2_rid_al, SB_T, 2.0) if chi2_rid_al(SB_T) > 0 else SB_T
for d in cs:
    d["s_c"] = sigma_c(d, ST)

ris_c = {}
for k in NOMI:
    for tutte in (False, True):
        sel = [d for d in cs if d["k"] == k and (tutte or not d["escl"])]
        if tutte and len(sel) == len([d for d in cs if d["k"] == k and not d["escl"]]):
            continue
        m, s, chi2, ndf, w = media_pesata([d["c"] for d in sel], [d["s_c"] for d in sel])
        syst = float(np.sum(w * np.array([d["dc_dMe"] for d in sel]))) * S_ME
        cv = np.array([d["c"] for d in sel])
        ris_c[(k, tutte)] = dict(
            N=len(sel), media=m, s_stat=s, s_syst=syst, s_tot=sqrt(s**2 + syst**2),
            chi2=chi2, ndf=ndf, p=p_chi2(chi2, ndf),
            media_aritm=cv.mean(), s_disp=cv.std(ddof=1) if len(cv) > 1 else float("nan"))

# alluminio: confronto con la sola dispersione (incertezza di tipo A)
for tutte in (False, True):
    r = ris_c[("Al", tutte)]
    r["sA"] = r["s_disp"] / sqrt(r["N"])


def z(v, s, ref):
    return abs(v - ref) / s


# --------------------------------------------------------------------------
# Calore latente, eq. (4) della scheda
# --------------------------------------------------------------------------
cl = []
for r in leggi("calore_latente.csv"):
    n = int(r["prova"])
    W1, W2, W3 = (float(r[c]) for c in ("thermos_vuoto_g", "thermos_acqua_g",
                                         "thermos_acqua_ghiaccio_g"))
    Tg, Ta, Tf = (float(r[c]) for c in ("Tg_C", "Ta_C", "Tf_C"))
    Tf = TF_QUADERNO.get(n, Tf)
    cl.append(dict(n=n, W1=W1, W2=W2, W3=W3, ma=W2 - W1, mg=W3 - W2, Tg=Tg, Ta=Ta, Tf=Tf,
                   escl=n in ESCLUSE_L))


def lam(d, me=ME):
    return CA * ((d["ma"] + me) * (d["Ta"] - d["Tf"]) - d["mg"] * (d["Tf"] - d["Tg"])) / d["mg"]


def sigma_lam(d, me=ME, sT=None):
    """Incertezza statistica di lambda per una prova (M_e fissata).
    m_a = W2 - W1 e m_g = W3 - W2 hanno in comune la pesata W2: covarianza -sigma_W^2."""
    sT = ST if sT is None else sT
    ma, mg, Ta, Tf = d["ma"], d["mg"], d["Ta"], d["Tf"]
    d_ma = CA * (Ta - Tf) / mg
    d_mg = -CA * (ma + me) * (Ta - Tf) / mg**2
    d_Ta = CA * (ma + me) / mg
    d_Tf = -CA * (ma + me + mg) / mg
    d_Tg = CA
    var_m = 2 * SB_M**2
    sTf = sT
    v = (d_ma**2 * var_m + d_mg**2 * var_m + 2 * d_ma * d_mg * (-SB_M**2)
         + d_Ta**2 * sT**2 + d_Tf**2 * sTf**2 + d_Tg**2 * sT**2)
    return sqrt(v)


for d in cl:
    d["lam"] = lam(d)
    d["s_lam"] = sigma_lam(d)
    d["dl_dMe"] = CA * (d["Ta"] - d["Tf"]) / d["mg"]
    d["x"] = (d["Ta"] - d["Tf"]) / d["mg"]
    d["y"] = lam(d, me=0.0) / CA

ris_l = {}
for tutte in (False, True):
    sel = [d for d in cl if tutte or not d["escl"]]
    m, s, chi2, ndf, w = media_pesata([d["lam"] for d in sel], [d["s_lam"] for d in sel])
    fs = sqrt(chi2 / ndf) if chi2 > ndf else 1.0    # fattore di scala (chi quadro rid.)
    syst = float(np.sum(w * np.array([d["dl_dMe"] for d in sel]))) * S_ME
    lv = np.array([d["lam"] for d in sel])
    ris_l[tutte] = dict(N=len(sel), media=m, s_stat0=s, fs=fs, s_stat=s * fs, s_syst=syst,
                        s_tot=sqrt((s * fs) ** 2 + syst**2), chi2=chi2, ndf=ndf,
                        p=p_chi2(chi2, ndf), media_aritm=lv.mean(), s_disp=lv.std(ddof=1))

# --------------------------------------------------------------------------
# Massa equivalente dalle prove di calore latente, senza valori esterni:
# lambda_g/c_a = y_i + M_e x_i  ->  y = a + b x, con a = lambda_g/c_a e b = -M_e
# --------------------------------------------------------------------------
sel = [d for d in cl if not d["escl"]]
me_fit = ME
for _ in range(20):
    sy = [sigma_lam(d, me=me_fit) / CA for d in sel]   # varianza del residuo
    a, b, sa, sb, cov_ab, chi2_fit, ndf_fit = fit_lineare([d["x"] for d in sel],
                                                         [d["y"] for d in sel], sy)
    if abs(-b - me_fit) < 1e-9:
        break
    me_fit = -b
fs_fit = sqrt(chi2_fit / ndf_fit) if chi2_fit > ndf_fit else 1.0
fit = dict(lam=a * CA, s_lam=sa * CA * fs_fit, Me=-b, s_Me=sb * fs_fit,
           rho=-cov_ab / (sa * sb), chi2=chi2_fit, ndf=ndf_fit, p=p_chi2(chi2_fit, ndf_fit),
           fs=fs_fit, s_lam0=sa * CA, s_Me0=sb, sy=sy,
           xmin=min(d["x"] for d in sel), xmax=max(d["x"] for d in sel))


# Massa equivalente "di taratura" con un valore di riferimento (non e' una misura
# indipendente: usa lambda_g o c_Al tabulati)
def me_da_lambda(d, lref=LAMBDA_REF):
    return d["mg"] * (lref / CA + d["Tf"] - d["Tg"]) / (d["Ta"] - d["Tf"]) - d["ma"]


def s_me_da_lambda(d):
    # derivate numeriche rispetto alle grandezze misurate
    base = dict(d)
    sT, sTf = ST, ST
    var = 0.0
    for chiave, s in (("Ta", sT), ("Tf", sTf), ("Tg", sT)):
        e = dict(base)
        e[chiave] += 1e-6
        var += ((me_da_lambda(e) - me_da_lambda(base)) / 1e-6 * s) ** 2
    # masse: W1, W2, W3 indipendenti
    for i, s in (("W1", SB_M), ("W2", SB_M), ("W3", SB_M)):
        e = dict(base)
        e[i] += 1e-6
        e["ma"], e["mg"] = e["W2"] - e["W1"], e["W3"] - e["W2"]
        var += ((me_da_lambda(e) - me_da_lambda(base)) / 1e-6 * s) ** 2
    return sqrt(var)


def me_da_al(d, cref=C_REF["Al"]):
    return cref * d["m1"] * d["dT2"] / (CA * d["dT1"]) - d["ma"]


def s_me_da_al(d):
    base = dict(d)
    var = 0.0
    for chiave, s in (("T1", ST), ("Ta", ST), ("Tf", ST)):
        e = dict(base)
        e[chiave] += 1e-6
        e["dT1"], e["dT2"] = e["Tf"] - e["Ta"], e["T1"] - e["Tf"]
        var += ((me_da_al(e) - me_da_al(base)) / 1e-6 * s) ** 2
    var += (C_REF["Al"] * d["dT2"] / (CA * d["dT1"]) * d["s_m1"]) ** 2
    var += d["s_ma"] ** 2
    return sqrt(var)


tar = {}
for nome, righe, f, sf in (("B", sel, me_da_lambda, s_me_da_lambda), ("A", al, me_da_al, s_me_da_al)):
    v = [f(d) for d in righe]
    s = [sf(d) for d in righe]
    m, sm, chi2, ndf, _ = media_pesata(v, s)
    fs = sqrt(chi2 / ndf) if chi2 > ndf else 1.0
    tar[nome] = dict(valori=v, sigma=s, media=m, s=sm * fs, s0=sm, chi2=chi2, ndf=ndf,
                     p=p_chi2(chi2, ndf), fs=fs)


# --------------------------------------------------------------------------
# Formattazione: incertezze con due cifre significative
# --------------------------------------------------------------------------
def arrotonda(v, s, nsig=2):
    e = floor(log10(abs(s)))
    dec = nsig - 1 - e
    sr = round(s, dec)
    if sr >= 10 ** (e + 1):
        dec -= 1
        sr = round(s, dec)
    dec_out = max(dec, 0)
    return f"{round(v, dec):.{dec_out}f}", f"{sr:.{dec_out}f}"


def decimali(s, nsig=2):
    a = arrotonda(1.0, s, nsig)[1]
    return len(a.split(".")[1]) if "." in a else 0


def con_dec(v, dec):
    return f"{round(v, dec):.{dec}f}"


def pm(v, s, nsig=2):
    a, b = arrotonda(v, s, nsig)
    return f"{a} +- {b}"


def cifre(v, s, nsig=2):
    return arrotonda(v, s, nsig)[0]


def f1(v, dec=1):
    return f"{v:.{dec}f}"


macro = {}


def M(nome, valore):
    macro[nome] = valore


M("sigmaTris", f"{SB_T:.2f}")
M("sigmaMris", f"{SB_M:.2f}")
M("sigmaTeff", f"{ST:.2f}")
M("rapportoST", f"{ST / SB_T:.1f}")
M("chiAlRis", f"{chi2_al_ris:.0f}")
M("chiAlRisRid", f"{chi2_al_ris / 3:.0f}")
for (k, tutte), r in ris_c.items():
    suf = k + ("Tutte" if tutte else "")
    dd = decimali(r["s_tot"])
    M(f"c{suf}", pm(r["media"], r["s_tot"]))
    M(f"c{suf}stat", f"{con_dec(r['media'], dd)} +- {con_dec(r['s_stat'], dd)}")
    M(f"c{suf}syst", con_dec(r["s_syst"], dd))
    M(f"c{suf}statonly", con_dec(r["s_stat"], dd))
    M(f"c{suf}J", pm(r["media"] * J_CAL, r["s_tot"] * J_CAL))
    M(f"c{suf}chi", f"{r['chi2']:.1f}")
    M(f"c{suf}ndf", f"{r['ndf']}")
    M(f"c{suf}p", fmt_p(r["p"]))
    M(f"c{suf}N", f"{r['N']}")
    M(f"c{suf}nu", f"{100 * r['s_tot'] / r['media']:.0f}")
for tutte in (False, True):
    suf = "Tutte" if tutte else ""
    r = ris_c[("Al", tutte)]
    M(f"cAl{suf}A", pm(r["media_aritm"], r["sA"]))
    M(f"cAl{suf}disp", cifre(r["s_disp"], r["s_disp"] / 10))
    M(f"cAl{suf}dispRel", f"{100 * r['s_disp'] / r['media_aritm']:.0f}")
    M(f"zAl{suf}", f"{z(r['media'], r['s_tot'], C_REF['Al']):.1f}")
M("zOtt", f"{z(ris_c[('Ott', False)]['media'], ris_c[('Ott', False)]['s_tot'], C_REF['Ott']):.1f}")
for k in ("RB", "RG"):
    r = ris_c[(k, False)]
    M(f"z{k}gra", f"{z(r['media'], r['s_tot'], C_REF['granito']):.1f}")
    M(f"z{k}qua", f"{z(r['media'], r['s_tot'], C_REF['quarzo']):.1f}")
    M(f"z{k}rocce", f"{max(0.0, (r['media'] - ROCCE[1]) / r['s_tot'], (ROCCE[0] - r['media']) / r['s_tot']):.1f}")
M("rocceMin", f"{ROCCE[0]:.2f}")
M("rocceMax", f"{ROCCE[1]:.2f}")
d_al1 = [d for d in cs if d["k"] == "Al" and d["n"] == 1][0]
r_al = ris_c[("Al", False)]
M("cAlUno", cifre(d_al1["c"], d_al1["s_c"]))
M("cAlUnoS", arrotonda(d_al1["c"], d_al1["s_c"])[1])
M("zAlUno", f"{abs(d_al1['c'] - r_al['media_aritm']) / r_al['s_disp']:.1f}")
for tutte in (False, True):
    suf = "Tutte" if tutte else ""
    r = ris_l[tutte]
    dd = decimali(r["s_tot"])
    M(f"lam{suf}", pm(r["media"], r["s_tot"]))
    M(f"lam{suf}stat", con_dec(r["s_stat"], dd))
    M(f"lam{suf}statzero", arrotonda(r["media"], r["s_stat0"])[1])
    M(f"lam{suf}syst", con_dec(r["s_syst"], dd))
    M(f"lam{suf}J", pm(r["media"] * J_CAL, r["s_tot"] * J_CAL))
    M(f"lam{suf}chi", f"{r['chi2']:.1f}")
    M(f"lam{suf}ndf", f"{r['ndf']}")
    M(f"lam{suf}p", fmt_p(r["p"]))
    M(f"lam{suf}fs", f"{r['fs']:.1f}")
    M(f"zlam{suf}", f"{z(r['media'], r['s_tot'], LAMBDA_REF):.1f}")
    M(f"lam{suf}scarto", f"{100 * abs(r['media'] - LAMBDA_REF) / LAMBDA_REF:.1f}")
d_l1 = [d for d in cl if d["n"] == 1][0]
M("lamUno", f1(d_l1["lam"]))
# prova 1 del calore latente: T_f o acqua liquida che darebbero il valore tabulato
tf_l1 = (((d_l1["ma"] + ME) * d_l1["Ta"] + d_l1["mg"] * d_l1["Tg"] - d_l1["mg"] * LAMBDA_REF / CA)
         / (d_l1["ma"] + ME + d_l1["mg"]))
M("TfLamUno", f1(tf_l1))
M("acquaLamUno", f"{d_l1['mg'] * (1 - d_l1['lam'] / LAMBDA_REF):.0f}")
M("mgLamUno", f1(d_l1["mg"]))
# prova 1 dell'alluminio: T_f che darebbe il valore tabulato
k_al1 = C_REF["Al"] * d_al1["m1"] / (CA * (d_al1["ma"] + ME))
tf_al1 = (d_al1["Ta"] + k_al1 * d_al1["T1"]) / (1 + k_al1)
M("TfAlUno", f1(tf_al1))
M("dTfAlUno", f1(tf_al1 - d_al1["Tf"]))
M("lamUnoS", arrotonda(d_l1["lam"], d_l1["s_lam"])[1])
M("lamUnoC", cifre(d_l1["lam"], d_l1["s_lam"]))  # con le cifre dell'incertezza
M("MeFit", pm(fit["Me"], fit["s_Me"]))
M("MeFitzero", pm(fit["Me"], fit["s_Me0"]))
M("lamFit", pm(fit["lam"], fit["s_lam"]))
M("rhoFit", f"{fit['rho']:.2f}")
M("chiFit", f"{fit['chi2']:.1f}")
M("pFit", fmt_p(fit["p"]))
M("fsFit", f"{fit['fs']:.1f}")
M("zMeFit", f"{abs(fit['Me'] - ME) / sqrt(fit['s_Me']**2 + S_ME**2):.1f}")
M("rapportoSMe", f"{fit['s_Me'] / S_ME:.0f}")
M("xmin", f"{fit['xmin']:.2f}")
M("xmax", f"{fit['xmax']:.2f}")
M("MeTarB", pm(tar["B"]["media"], tar["B"]["s"]))
M("MeTarBfs", f"{tar['B']['fs']:.1f}")
M("MeTarBchi", f"{tar['B']['chi2']:.1f}")
# z dai valori arrotondati riportati nel testo, cosi' il lettore lo ritrova con quei valori
_mB, _sB = (float(x) for x in arrotonda(tar["B"]["media"], tar["B"]["s"]))
M("zMeTarB", f"{abs(_mB - ME) / sqrt(_sB**2 + S_ME**2):.1f}")
M("MeTarA", pm(tar["A"]["media"], tar["A"]["s"]))
M("MeTarAfs", f"{tar['A']['fs']:.1f}")
M("MeTarAchi", f"{tar['A']['chi2']:.1f}")
_mA, _sA = (float(x) for x in arrotonda(tar["A"]["media"], tar["A"]["s"]))
M("zMeTarA", f"{abs(_mA - ME) / sqrt(_sA**2 + S_ME**2):.1f}")
# acqua trascinata con il ghiaccio che spiegherebbe lo scarto di lambda
r = ris_l[False]
mg_medio = np.mean([d["mg"] for d in sel])
f_acqua = 1 - r["media"] / LAMBDA_REF
M("fracAcqua", f"{100 * f_acqua:.0f}")
M("gAcqua", f"{f_acqua * mg_medio:.0f}")
M("mgMedio", f"{mg_medio:.0f}")
# rocce: rapporto tra capacita' termica del campione e dell'acqua
for k in ("RB", "RG", "Al", "Ott"):
    sel_k = [d for d in cs if d["k"] == k and not d["escl"]]
    M(f"dTa{k}", f"{np.mean([d['dT1'] for d in sel_k]):.1f}")

# acqua calda trasportata che spiegherebbe lo scarto della roccia grigia dal granito
M("acquaRG", f"{(ris_c[('RG', False)]['media'] - C_REF['granito']) * masse['RG'][0] / CA:.1f}")
# capacita' termiche delle rocce e dell'acqua con il thermos
for k in ("RB", "RG"):
    M(f"C{k}", f"{masse[k][0] * ris_c[(k, False)]['media']:.0f}")
M("CacquaMin", f"{min(d['ma'] for d in cs if d['k'] in ('RB', 'RG')) + ME:.0f}")
M("CacquaMax", f"{max(d['ma'] for d in cs if d['k'] in ('RB', 'RG')) + ME:.0f}")
# colonna c_1 del logbook: il thermos vuoto non veniva sottratto (macro \mthermos1)
rapp = [((d["W"] - 1 + ME) / (d["ma"] + ME)) for d in cs]
M("rappLogMin", f"{min(rapp):.1f}")
M("rappLogMax", f"{max(rapp):.1f}")
M("chiAlRisP", fmt_p(p_chi2(chi2_al_ris, 3)))
dl_dTf = [CA * (d["ma"] + ME + d["mg"]) / d["mg"] for d in cl if not d["escl"]]
M("dlTfMin", f"{min(dl_dTf):.1f}")
M("dlTfMax", f"{max(dl_dTf):.1f}")

# --------------------------------------------------------------------------
# Numeri aggiunti per la versione del 14/10/2026 (esempi di calcolo e spiegazioni)
# --------------------------------------------------------------------------
# esempio di calcolo: alluminio, prova 2
ex = [d for d in cs if d["k"] == "Al" and d["n"] == 2][0]
M("exAlPa", f1(ex["W"]))
M("exAlMa", f1(ex["ma"]))
M("exAlMaMe", f1(ex["ma"] + ME))
M("exAlTuno", f1(ex["T1"]))
M("exAlTa", f1(ex["Ta"]))
M("exAlTf", f1(ex["Tf"]))
M("exAlDTa", f1(ex["dT1"]))
M("exAlDTc", f1(ex["dT2"]))
M("exAlNum", f1(CA * (ex["ma"] + ME) * ex["dT1"]))
M("exAlDen", f1(ex["m1"] * ex["dT2"]))
M("exAlC", f"{ex['c']:.4f}")
M("exAlCr", cifre(ex["c"], ex["s_c"]))
M("exAlS", arrotonda(ex["c"], ex["s_c"])[1])
M("exAlpTa", f"{100 * ST / ex['dT1']:.1f}")
M("exAlpTc", f"{100 * ST / ex['dT2']:.1f}")
M("exAlpTf", f"{100 * ST * (1 / ex['dT1'] + 1 / ex['dT2']):.1f}")
M("exAlpTot", f"{100 * ex['s_c'] / ex['c']:.1f}")
M("exAlpRis", f"{100 * ex['s_c_ris'] / ex['c']:.1f}")
# incertezze relative delle prove valide dell'alluminio, con la sola risoluzione e con sigma_T
nu_ris = [100 * d["s_c_ris"] / d["c"] for d in al]
nu_eff = [100 * d["s_c"] / d["c"] for d in al]
M("nuRisMin", f"{min(nu_ris):.0f}")
M("nuRisMax", f"{max(nu_ris):.0f}")
M("nuEffMin", f"{min(nu_eff):.0f}")
M("nuEffMax", f"{max(nu_eff):.0f}")
M("cAlMin", f"{min(d['c'] for d in al):.3f}")
M("cAlMax", f"{max(d['c'] for d in al):.3f}")
M("cAldispDue", arrotonda(r_al["s_disp"], r_al["s_disp"])[1])
# se si applicasse direttamente la dispersione relativa dell'alluminio agli altri campioni
rel_al = r_al["s_disp"] / r_al["media_aritm"]
for k in ("Ott", "RB", "RG"):
    v = np.array([d["c"] for d in cs if d["k"] == k and not d["escl"]])
    s12 = media_pesata(v, rel_al * v)[1]
    M(f"sDodici{k}", arrotonda(s12, s12)[1])
    M(f"rappDodici{k}", f"{ris_c[(k, False)]['s_stat'] / s12:.1f}")
# roccia grigia: acqua calda trattenuta che porterebbe c nell'intervallo delle rocce comuni
c_rg, m_rg = ris_c[("RG", False)]["media"], masse["RG"][0]
mw_min, mw_max = (c_rg - ROCCE[1]) * m_rg / CA, (c_rg - ROCCE[0]) * m_rg / CA
M("acquaRGmin", f"{mw_min:.0f}")
M("acquaRGmax", f"{mw_max:.0f}")
M("acquaRGpmin", f"{100 * mw_min / m_rg:.0f}")
M("acquaRGpmax", f"{100 * mw_max / m_rg:.0f}")
M("mRG", f1(m_rg))
# esempio di calcolo: calore latente, prova 2
el = [d for d in cl if d["n"] == 2][0]
M("exLPa", f1(el["W2"]))
M("exLPag", f1(el["W3"]))
M("exLMa", f1(el["ma"]))
M("exLMaMe", f1(el["ma"] + ME))
M("exLMg", f1(el["mg"]))
M("exLTg", f"{el['Tg']:.2f}")
M("exLTa", f1(el["Ta"]))
M("exLTf", f1(el["Tf"]))
M("exLDTa", f1(el["Ta"] - el["Tf"]))
M("exLDTg", f1(el["Tf"] - el["Tg"]))
M("exLNumUno", f1((el["ma"] + ME) * (el["Ta"] - el["Tf"])))
M("exLNumDue", f1(el["mg"] * (el["Tf"] - el["Tg"])))
M("exLlam", f"{el['lam']:.2f}")
M("exLlamr", cifre(el["lam"], el["s_lam"]))
M("exLS", arrotonda(el["lam"], el["s_lam"])[1])
d_tf = CA * (el["ma"] + ME + el["mg"]) / el["mg"]
M("exLdTf", f"{d_tf:.2f}")
M("exLcTf", f"{d_tf * ST:.1f}")
M("exLdTa", f"{CA * (el['ma'] + ME) / el['mg']:.2f}")
M("exLcTa", f"{CA * (el['ma'] + ME) / el['mg'] * ST:.1f}")
# calore latente: media aritmetica con l'incertezza di tipo A (confronto con il fattore di scala)
rl = ris_l[False]
M("lamA", pm(rl["media_aritm"], rl["s_disp"] / sqrt(rl["N"])))
M("lamsigmaA", arrotonda(rl["media_aritm"], rl["s_disp"] / sqrt(rl["N"]))[1])
# scarti delle prove dal valore tabulato
for d in cl:
    M("scartoL" + ["", "Uno", "Due", "Tre", "Quattro"][d["n"]],
      f"{100 * (d['lam'] - LAMBDA_REF) / LAMBDA_REF:.1f}")
# di quanto dovrebbe essere maggiore M_e del thermos B per spiegare lo scarto di lambda
sel_l = [d for d in cl if not d["escl"]]
w_l = np.array([1 / d["s_lam"] ** 2 for d in sel_l])
dl_me = float(np.sum(w_l * np.array([d["dl_dMe"] for d in sel_l])) / w_l.sum())
M("dlMe", f"{dl_me:.2f}")
M("dMeLam", f"{(LAMBDA_REF - rl['media']) / dl_me:.0f}")
# quante volte l'incertezza di M_e della scheda
M("zMeLam", f"{(LAMBDA_REF - rl['media']) / dl_me / S_ME:.1f}")
# probabilita' (coda di una gaussiana) che M_e superi il valore che spiegherebbe lo scarto, se
# l'incertezza della scheda e' una deviazione standard
p_me = 100 * norm.sf((LAMBDA_REF - rl["media"]) / dl_me / S_ME)
M("pMeLam", f"{p_me:.1f}" if p_me < 10 else f"{p_me:.0f}")
M("MeLam", f"{ME + (LAMBDA_REF - rl['media']) / dl_me:.0f}")
# letture di temperatura interpolate tra due tacche (decimi dispari con il termometro da 0.2 gradi)
interp = []
for d in cs:
    for chiave in ("T1", "Ta", "Tf"):
        if round(d[chiave] * 10) % 2 == 1:
            interp.append(d[chiave])
for d in cl:
    for chiave in ("Ta", "Tf") + (("Tg",) if d["n"] != 1 else ()):
        if round(d[chiave] * 10) % 2 == 1:
            interp.append(d[chiave])
lst = [f"\\num{{{v:.1f}}}" for v in interp]
M("letInterp", ", ".join(lst[:-1]) + " e " + lst[-1])
M("nInterp", str(len(interp)))

# --------------------------------------------------------------------------
# Numeri aggiunti il 18/10/2026 (risposte ai commenti del gruppo sulla versione del 14/10)
# --------------------------------------------------------------------------
# sigma_T: con le masse trascurabili coincide con la regola del fattore di scala
M("chiAlRisUno", f"{chi2_al_ris:.1f}")
M("sigmaTrisTre", f"{SB_T:.3f}")
M("rapportoSTchi", f"{sqrt(chi2_al_ris / 3):.2f}")
M("sigmaTapprox", f"{SB_T * sqrt(chi2_al_ris / 3):.2f}")
M("sigmaTeffTre", f"{ST:.3f}")
# contributi delle masse all'incertezza relativa della prova 2 dell'alluminio
M("exAlpMa", f"{100 * ex['s_ma'] / (ex['ma'] + ME):.2f}")
M("exAlpMuno", f"{100 * ex['s_m1'] / ex['m1']:.2f}")
# alluminio, prova 1: z rispetto alla media pesata delle prove 2-5, con le due incertezze
# statistiche in quadratura (M_e e' comune e si cancella nella differenza), e aumento del
# chi quadro quando la si aggiunge (coincide con z^2)
r_al_t = ris_c[("Al", True)]
z_al1 = abs(d_al1["c"] - r_al["media"]) / sqrt(d_al1["s_c"] ** 2 + r_al["s_stat"] ** 2)
dchi_al1 = r_al_t["chi2"] - r_al["chi2"]
assert abs(dchi_al1 - z_al1 ** 2) < 1e-6 * dchi_al1, (dchi_al1, z_al1 ** 2)
M("zAlUnoComb", f"{z_al1:.1f}")
M("zAlUnoCombQ", f"{z_al1 ** 2:.1f}")
M("dchiAlUno", f"{dchi_al1:.1f}")
M("cAlUnoCtre", f"{d_al1['c']:.3f}")
# stessi numeri con quattro decimali, per i calcoli scritti per esteso nel testo: z ricalcolato
# dai valori arrotondati, cosi' il lettore ritrova le stesse cifre
q4 = lambda v: round(v, 4)
M("cAlUnoQ", f"{d_al1['c']:.4f}")
M("cAlUnoSQ", f"{d_al1['s_c']:.4f}")
M("cAlCQ", f"{r_al['media']:.4f}")
M("cAlStatQ", f"{r_al['s_stat']:.4f}")
M("cAlAritmQ", f"{r_al['media_aritm']:.4f}")
M("cAlSdispQ", f"{r_al['s_disp']:.4f}")
z_disp = abs(q4(d_al1["c"]) - q4(r_al["media"])) / sqrt(q4(d_al1["s_c"]) ** 2 + q4(r_al["s_stat"]) ** 2)
d_disp = abs(q4(d_al1["c"]) - q4(r_al["media_aritm"])) / q4(r_al["s_disp"])
assert f"{d_disp:.1f}" == f"{abs(d_al1['c'] - r_al['media_aritm']) / r_al['s_disp']:.1f}"
assert abs(z_disp - z_al1) < 0.02 and f"{z_disp ** 2:.1f}" == f"{dchi_al1:.1f}"
M("zAlUnoDisp", f"{z_disp:.2f}")

M("cAlCtre", f"{r_al['media']:.3f}")
M("cAlStatTre", f"{r_al['s_stat']:.3f}")
M("cAlUnoStre", f"{d_al1['s_c']:.3f}")
M("cAlAritmTre", f"{r_al['media_aritm']:.3f}")
M("cAlSdispTre", f"{r_al['s_disp']:.3f}")
# calore latente: valori delle prove valide, media pesata
lv = [d["lam"] for d in cl if not d["escl"]]
M("lamMin", f"{min(lv):.1f}")
M("lamMax", f"{max(lv):.1f}")
M("lamC", cifre(ris_l[False]["media"], ris_l[False]["s_tot"]))
M("lamstatzeroC", con_dec(ris_l[False]["media"], 2))
# calore latente con le T_f del logbook (arrotondate al grado), per la tabella delle differenze:
# incertezza di arrotondamento 1 grado/sqrt(12) in quadratura con sigma_T
cl_log = []
for r_ in leggi("calore_latente.csv"):
    n_ = int(r_["prova"])
    if n_ in ESCLUSE_L:
        continue
    W1, W2, W3 = (float(r_[c]) for c in ("thermos_vuoto_g", "thermos_acqua_g",
                                         "thermos_acqua_ghiaccio_g"))
    dd_ = dict(n=n_, ma=W2 - W1, mg=W3 - W2, Tg=float(r_["Tg_C"]), Ta=float(r_["Ta_C"]),
               Tf=float(r_["Tf_C"]))
    ma_, mg_, Ta_, Tf_ = dd_["ma"], dd_["mg"], dd_["Ta"], dd_["Tf"]
    sTf_ = sqrt(ST ** 2 + SB_TF_LOG ** 2)
    v_ = ((CA * (Ta_ - Tf_) / mg_) ** 2 * 2 * SB_M ** 2
          + (CA * (ma_ + ME) * (Ta_ - Tf_) / mg_ ** 2) ** 2 * 2 * SB_M ** 2
          + 2 * (CA * (Ta_ - Tf_) / mg_) * (-CA * (ma_ + ME) * (Ta_ - Tf_) / mg_ ** 2) * (-SB_M ** 2)
          + (CA * (ma_ + ME) / mg_) ** 2 * ST ** 2
          + (CA * (ma_ + ME + mg_) / mg_) ** 2 * sTf_ ** 2 + CA ** 2 * ST ** 2)
    dd_["lam"], dd_["s_lam"], dd_["dl_dMe"] = lam(dd_), sqrt(v_), CA * (Ta_ - Tf_) / mg_
    cl_log.append(dd_)
m_, s_, chi2_, ndf_, w_ = media_pesata([d["lam"] for d in cl_log], [d["s_lam"] for d in cl_log])
fs_ = sqrt(chi2_ / ndf_) if chi2_ > ndf_ else 1.0
syst_ = float(np.sum(w_ * np.array([d["dl_dMe"] for d in cl_log]))) * S_ME
M("lamLogbook", pm(m_, sqrt((s_ * fs_) ** 2 + syst_ ** 2)))
M("lamLogbookchi", f"{chi2_:.1f}")
# acqua liquida trasferita con il ghiaccio che spiegherebbe lo scarto di lambda
M("fracAcquaU", f"{100 * f_acqua:.1f}")
M("fracAcquaTre", f"{f_acqua:.3f}")
M("gAcquaU", f"{f_acqua * mg_medio:.1f}")
M("mgMedioU", f"{mg_medio:.1f}")
M("lamSuMg", f"{LAMBDA_REF / mg_medio:.2f}")
M("deltaLam", f"{LAMBDA_REF - r['media']:.2f}")
# massa equivalente che spiegherebbe lo scarto di lambda
M("dlMeTre", f"{dl_me:.3f}")
M("dMeLamU", f"{(LAMBDA_REF - rl['media']) / dl_me:.1f}")
M("MeLamU", f"{ME + (LAMBDA_REF - rl['media']) / dl_me:.1f}")
# con M_e all'estremo superiore dell'intervallo della scheda (25 + 5 g): lambda aumenta di
# 5 g x dl_me (lambda e' lineare in M_e; stessi pesi della eq. sist) e l'acqua liquida che
# spiegherebbe lo scarto residuo diminuisce (richiesta di Lennard, 18/10/2026)
dl_cinque = S_ME * dl_me
d_res = (LAMBDA_REF - rl["media"]) - dl_cinque
M("MeMaxScheda", f"{ME + S_ME:.0f}")
M("dlMeCinque", f"{dl_cinque:.2f}")
M("deltaLamMax", f"{d_res:.2f}")
M("fracAcquaMaxU", f"{100 * d_res / LAMBDA_REF:.1f}")
M("fracAcquaMaxTre", f"{d_res / LAMBDA_REF:.3f}")
M("gAcquaMaxU", f"{d_res / LAMBDA_REF * mg_medio:.1f}")
# roccia grigia: acqua trattenuta, con i limiti dell'intervallo a tre decimali
M("rocceMinTre", f"{ROCCE[0]:.3f}")
M("rocceMaxTre", f"{ROCCE[1]:.3f}")
M("cRGtre", f"{c_rg:.3f}")
M("acquaRGminU", f"{mw_min:.1f}")
M("acquaRGmaxU", f"{mw_max:.1f}")
M("acquaRGpminU", f"{100 * round(mw_min, 1) / m_rg:.0f}")
M("acquaRGpmaxU", f"{100 * round(mw_max, 1) / m_rg:.0f}")
# parte facoltativa: incertezza del fit, stime con i valori tabulati prova per prova
M("sMeFit", arrotonda(fit["Me"], fit["s_Me"])[1])
M("MeAvalori", "\\qtylist{" + ";".join(f"{v:.0f}" for v in tar["A"]["valori"]) + "}{\\gram}")
M("MeAsMin", f"{min(tar['A']['sigma']):.0f}")
M("MeAsMax", f"{max(tar['A']['sigma']):.0f}")
M("MeBvalori", "\\qtylist{" + ";".join(f"{v:.0f}" for v in tar["B"]["valori"]) + "}{\\gram}")
M("MeBsMin", f"{min(tar['B']['sigma']):.1f}")
M("MeBsMax", f"{max(tar['B']['sigma']):.1f}")
ex_me = C_REF["Al"] * ex["m1"] * ex["dT2"] / (CA * ex["dT1"])
M("exMeAuno", f1(ex_me))
M("exMeA", f1(ex_me - ex["ma"]))
el_me = el["mg"] * (LAMBDA_REF / CA + el["Tf"] - el["Tg"]) / (el["Ta"] - el["Tf"])
M("exMeBuno", f1(el_me))
M("exMeB", f1(el_me - el["ma"]))
M("exMeBsomma", f1(LAMBDA_REF / CA + el["Tf"] - el["Tg"]))

with open(GEN / "numeri.tex", "w", encoding="utf-8") as f:
    f.write("% Generato da analisi_PP1.py: non modificare a mano\n")
    for k, v in macro.items():
        f.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")

# --------------------------------------------------------------------------
# Tabelle
# --------------------------------------------------------------------------
with open(GEN / "tab_masse.tex", "w", encoding="utf-8") as f:
    for k in NOMI:
        m, s, p = masse[k]
        f.write(f"{NOMI[k]} & {p[0]:.1f} & {p[1]:.1f} & {p[2]:.1f} & {m:.1f} & {s:.2f} \\\\\n")

with open(GEN / "tab_cs.tex", "w", encoding="utf-8") as f:
    prec = None
    for d in cs:
        if prec is not None and d["k"] != prec:
            f.write("\\midrule\n")
        prec = d["k"]
        nome = f"{NOMI[d['k']]} {d['n']}" + ("$^{*}$" if d["escl"] else "")
        f.write(f"{nome} & {d['W']:.1f} & {d['ma']:.1f} & {d['T1']:.1f} & {d['Ta']:.1f} & "
                f"{d['Tf']:.1f} & {d['dT1']:.1f} & {d['dT2']:.1f} & \\num{{{pm(d['c'], d['s_c'])}}} \\\\\n")

with open(GEN / "tab_cl.tex", "w", encoding="utf-8") as f:
    for d in cl:
        nome = f"{d['n']}" + ("$^{*}$" if d["escl"] else "")
        tg = {"0.05": "0.05", "0.00": "0.00"}.get(f"{d['Tg']:.2f}", f"{d['Tg']:.1f}")
        f.write(f"{nome} & {d['W2']:.1f} & {d['W3']:.1f} & {d['ma']:.1f} & {d['mg']:.1f} & "
                f"{tg} & {d['Ta']:.1f} & {d['Tf']:.1f} & \\num{{{pm(d['lam'], d['s_lam'])}}} \\\\\n")

with open(GEN / "tab_risultati.tex", "w", encoding="utf-8") as f:
    rif = {"Al": f"{C_REF['Al']:.3f}", "Ott": f"{C_REF['Ott']:.3f}",
           "RB": f"{C_REF['granito']:.2f}$^{{a}}$",
           "RG": "---"}  # natura ignota: nessun riferimento
    for k in NOMI:
        r = ris_c[(k, False)]
        dd = decimali(r["s_tot"])
        a, s_st, s_sy, s_t = (con_dec(v, dd) for v in (r["media"], r["s_stat"],
                                                        r["s_syst"], r["s_tot"]))
        if k in ("Al", "Ott"):
            zz = f"{z(r['media'], r['s_tot'], C_REF[k]):.1f}"
        elif k == "RB":
            zz = f"{z(r['media'], r['s_tot'], C_REF['granito']):.1f}"
        else:
            zz = "---"
        f.write(f"{NOMI[k]} & {r['N']} & {a} & {s_st} & {s_sy} & {s_t} & "
                f"{r['chi2']:.1f}/{r['ndf']} & {rif[k]} & {zz} \\\\\n")

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 0.9, "errorbar.capsize": 2, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
BLU, ARANCIO, GRIGIO = "#1f5fa8", "#c8501e", "#555555"
CM = 1 / 2.54

fig, assi = plt.subplots(2, 2, figsize=(15 * CM, 7.6 * CM))
for ax, k in zip(assi.flat, NOMI):
    sel_k = [d for d in cs if d["k"] == k]
    r = ris_c[(k, False)]
    ax.axhspan(r["media"] - r["s_tot"], r["media"] + r["s_tot"], color=BLU, alpha=0.15, lw=0,
               label="media pesata $\\pm\\,\\sigma$")
    ax.axhline(r["media"], color=BLU, lw=0.8)
    for d in sel_k:
        ax.errorbar(d["n"], d["c"], yerr=d["s_c"], fmt="o", ms=3.2,
                    color=GRIGIO if d["escl"] else "k",
                    mfc="white" if d["escl"] else "k", mew=0.8, elinewidth=0.8)
    if k in ("Al", "Ott"):
        ax.axhline(C_REF[k], color=ARANCIO, ls="--", lw=0.9,
                   label="valore tabulato" if k == "Al" else "ottone (tabulato)")
    else:
        ax.axhspan(*ROCCE, facecolor="none", edgecolor=ARANCIO, hatch="////", lw=0,
                   alpha=0.45, label="rocce comuni")
        if k == "RB":  # la roccia grigia è di natura ignota: niente valore tabulato
            ax.axhline(C_REF["granito"], color=ARANCIO, ls="--", lw=0.9, label="granito")
    if k == "RB":
        ax.set_ylim(0.05, 0.235)
    if k == "RG":
        ax.set_ylim(0.13, 0.6)
    ax.set_title(NOMI[k])
    ax.set_xticks([d["n"] for d in sel_k])
    ax.set_xlim(0.5, len(sel_k) + 0.5)
    ax.set_xlabel("prova")
    ax.set_ylabel("$c_1$ [cal/(g K)]")
    ax.legend(loc={"Al": "lower right", "Ott": "lower center", "RB": "lower left",
                   "RG": "upper left"}[k], ncol=1, frameon=True, framealpha=0.9, edgecolor="none",
              handlelength=1.6, borderaxespad=0.3)
fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=0.6)
fig.savefig(FIG / "calore_specifico.pdf")
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(8.0 * CM, 5.0 * CM))
r = ris_l[False]
ax1.axhspan(r["media"] - r["s_tot"], r["media"] + r["s_tot"], color=BLU, alpha=0.15, lw=0,
            label="media pesata $\\pm\\,\\sigma$ (prove 2-4)")
ax1.axhline(r["media"], color=BLU, lw=0.8)
ax1.axhline(LAMBDA_REF, color=ARANCIO, ls="--", lw=0.9, label="valore tabulato")
for d in cl:
    ax1.errorbar(d["n"], d["lam"], yerr=d["s_lam"], fmt="o", ms=3.2,
                 color=GRIGIO if d["escl"] else "k", mfc="white" if d["escl"] else "k",
                 mew=0.8, elinewidth=0.8)
ax1.set_xticks([d["n"] for d in cl])
ax1.set_xlim(0.5, len(cl) + 0.5)
ax1.set_xlabel("prova")
ax1.set_ylabel("$\\lambda_g$ [cal/g]")
ax1.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none",
           handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "calore_latente.pdf")
plt.close(fig)

# parte facoltativa: fit y = lambda_g/c_a - M_e x sulle prove valide di calore latente
fig, ax2 = plt.subplots(figsize=(8.0 * CM, 5.2 * CM))
xs = np.array([d["x"] for d in sel])
ys = np.array([d["y"] for d in sel])
xx = np.linspace(0.2, 0.5, 200)
yfit = fit["lam"] / CA - fit["Me"] * xx
# banda a 1 sigma della retta (con la correlazione tra i parametri, incertezze scalate)
sa_, sb_ = fit["s_lam"] / CA, fit["s_Me"]
banda = np.sqrt(np.clip(sa_**2 + (xx * sb_) ** 2 - 2 * xx * fit["rho"] * sa_ * sb_, 0, None))
ax2.fill_between(xx, yfit - banda, yfit + banda, color=BLU, alpha=0.15, lw=0)
ax2.plot(xx, yfit, color=BLU, lw=0.9,
         label=f"fit: $M_e$ = {arrotonda(fit['Me'], fit['s_Me'])[0]} g")
ax2.plot(xx, LAMBDA_REF / CA - ME * xx, color=ARANCIO, ls="--", lw=0.9,
         label="$\\lambda_g$ tabulato, $M_e=25$ g")
ax2.errorbar(xs, ys, yerr=fit["sy"], fmt="o", ms=3.2, color="k", mew=0.8, elinewidth=0.8)
for d in sel:
    ax2.annotate(str(d["n"]), (d["x"], d["y"]), xytext=(4, 3), textcoords="offset points",
                 fontsize=6.5)
ax2.set_xlim(0.2, 0.5)
ax2.set_xlabel("$x=(T_a-T_f)/m_g$ [K/g]")
ax2.set_ylabel("$y$ [K]")
ax2.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none",
           handlelength=1.6)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "massa_equivalente.pdf")
plt.close(fig)

# --------------------------------------------------------------------------
# Riepilogo a terminale
# --------------------------------------------------------------------------
print(f"sigma_T risoluzione = {SB_T:.3f}; chi2 Al con risoluzione = {chi2_al_ris:.1f}/3;"
      f" sigma_T efficace = {ST:.3f}")
for d in cs:
    print(f"{d['k']:3s}{d['n']} ma={d['ma']:6.1f} dT1={d['dT1']:4.1f} dT2={d['dT2']:4.1f} "
          f"c={d['c']:.4f} s_ris={d['s_c_ris']:.4f} s={d['s_c']:.4f}{' (esclusa)' if d['escl'] else ''}")
for (k, t), r in ris_c.items():
    print(k, "tutte" if t else "", {kk: round(v, 5) if isinstance(v, float) else v for kk, v in r.items()})
for d in cl:
    print(f"lat{d['n']} ma={d['ma']:.1f} mg={d['mg']:.1f} lam={d['lam']:.2f} s={d['s_lam']:.2f} "
          f"dl/dMe={d['dl_dMe']:.3f} x={d['x']:.4f} y={d['y']:.2f}")
for t, r in ris_l.items():
    print("lambda", "tutte" if t else "", {kk: round(v, 4) if isinstance(v, float) else v for kk, v in r.items()})
print("fit", {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in fit.items() if kk != 'sy'})
for nome, r in tar.items():
    print("taratura", nome, {kk: (np.round(v, 2) if not isinstance(v, int) else v) for kk, v in r.items()})
