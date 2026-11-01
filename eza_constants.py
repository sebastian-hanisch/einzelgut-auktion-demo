"""Konstanten: Regler-Grenzen, Messparameter der vorgerechneten Reihen, Farben und Presets.

Der Teil bis zur Hash-Marke (die Kommentarzeile „ab hier nicht mehr im Erzeuger-Hash“, `eza_experiments.HASH_END`) gehört zum Erzeuger-Hash der Messreihen in `data/`
(siehe `eza_experiments.py`); Regler, Presets, Farben und Hilfetexte dahinter ändern den Hash nicht. Komplexität an den Reglermaxima: Gitterrechnung O(GRID_N · n), Hülle O(HULL_N),
Lernlauf O(Runden · n · 21); ein Lauf mit 30.000 Runden und 4 Bietern dauert einige Sekunden (Knopf mit Spinner)."""

# --- Messparameter der vorgerechneten Reihen (Teil des Erzeuger-Hashs)
BLOCKS = 4                                            # unabhängige Blöcke (SplitMix64-Ströme) je Zelle
SIM_AUCTIONS = 200_000                                # simulierte Auktionen je Block
ASYM_SIM_AUCTIONS = 400_000
E1_NS = tuple(range(2, 11))                           # Erstpreis gegen Zweitpreis gegen All-pay, Gleichverteilung
E1_OTHER = (("exp", (2, 5)), ("beta", (2, 5)), ("mix", (2, 5)))
E1_GAP_VALUES = (20.0, 50.0, 80.0, 100.0)             # Werte, an denen die Beste-Antwort-Lücke geprüft wird (Euro)
E2_NS = (2, 5, 10)
E2_RHOS = (1.0, 0.75, 0.5, 0.25)
E3_V0S = (0.0, 20.0, 40.0)
E3_NS = (1, 2, 3, 5, 10)
E3_OTHER_V0S = (0.0, 20.0)
E3_OTHER_NS = (1, 2, 5)
E3_SIM_CELLS = ((2, 20.0, (50.0, 60.0, 70.0)), (5, 20.0, (50.0, 60.0, 70.0)))     # (n, v0, Mindestgebote) für die Simulation
E3_FPA_CELLS = ((2, 50.0), (3, 30.0), (5, 60.0))                                    # (n, r) für Erstpreis = Zweitpreis mit Mindestgebot
E4_NS = tuple(range(1, 11))
E4_MIX_NS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 60)
E4_V0 = 20.0                                          # Eigenwert in der zweiten Reihe ("serious bidder")
E4_MIX_EXTRA_NS = (1, 2, 3, 5, 10)
E5_WS = (1.0, 0.9, 0.75, 0.5, 0.25)
LEARN_BLOCKS = 4
LEARN_NS_TABLE = (2, 4)
LEARN_T = {"Hedge": 20_000, "Exp3": 100_000, "Q-Lernen": 300_000}
LEARN_Q_NS = (2,)                                     # Q-Lernen nur mit zwei Bietern (Laufzeit)
LEARN_HORIZONS = (2_000, 20_000, 100_000)             # Hedge, n = 2: Abhängigkeit vom Horizont
LEARN_FIXED_ETA = 1.0                                 # Hedge, n = 2: feste Rate (Vergleich mit der Standardrate)
LEARN_FIXED_ETA_HORIZONS = (2_000, 20_000)

# --- ab hier nicht mehr im Erzeuger-Hash

# --- Regler (Seitenleiste)
N_MIN, N_MAX, DEFAULT_N = 1, 20, 2                    # Zahl der Spediteure
R_MIN, R_MAX, DEFAULT_R = 0, 100, 0                   # Mindestgebot (Euro je Slot)
V0_MIN, V0_MAX, DEFAULT_V0 = 0, 60, 0                 # Eigenwert des Verkäufers (Euro je Slot)
RHO_MIN, RHO_MAX, DEFAULT_RHO = 0.25, 1.0, 1.0        # Risikoexponent u(x) = x^ρ (1 = risikoneutral)
W_MIN, W_MAX, DEFAULT_W = 0.25, 1.0, 1.0              # Schwäche des zweiten Bieters (Werte U[0, 100·w]); 1 = gleich stark
ROUNDS_MIN, ROUNDS_MAX, DEFAULT_ROUNDS = 1000, 30000, 5000
RATE_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 30.0)  # Lernrate als Vielfaches der Standardrate der Regel
DEFAULT_RATE = 1.0
LEARN_NS = (2, 4)
SEED_MAX, DEFAULT_SEED = 999999, 35
DIST_DEFAULT = "gleich"

COLOR_FIRST = "#c0504d"
COLOR_SECOND = "#2F6B65"
COLOR_ALLPAY = "#8064a2"
COLOR_BEST = "#d79b00"
COLOR_NEUTRAL = "rgba(150,150,150,0.8)"

PRESETS = {
    "Zwei Bieter": {"dist": "gleich", "n": 2},
    "Viele Bieter": {"dist": "gleich", "n": 10},
    "Reserve-Gipfel": {"dist": "gleich", "n": 2, "r": 60, "v0": 20},
    "Risikoscheu": {"dist": "gleich", "n": 2, "rho": 0.5},
    "Ein Bieter mehr statt Mindestgebot": {"dist": "gleich", "n": 1},
    "Unregulär (Bulow–Klemperer versagt)": {"dist": "mix", "n": 1},
    "Asymmetrisch": {"dist": "gleich", "n": 2, "w": 0.5},
    "Lernende": {"dist": "gleich", "n": 2, "rule": "Hedge", "fmt": "Erstpreis", "n_learn": 2, "rounds": 20000},
}
PRESET_HELP = {
    "Zwei Bieter": "Zwei Spediteure, Werte gleichverteilt zwischen 0 und 100 Euro, kein Mindestgebot: Erst-, Zweitpreis und All-pay bringen im Mittel 33,33 Euro.",
    "Viele Bieter": "Zehn Spediteure: der Spediteur mit Wert 50 Euro bietet im Erstpreis 45 Euro (bei zwei Bietern 25 Euro), der Erlös steigt auf 81,82 Euro.",
    "Reserve-Gipfel": "Zwei Bieter, Eigenwert des Verkäufers 20 Euro, Mindestgebot 60 Euro: das Maximum der Nutzenkurve, 47,73 Euro statt 37,07 Euro ohne Mindestgebot.",
    "Risikoscheu": "Zwei risikoscheue Bieter (Exponent 0,5): der Erstpreis bringt 44,44 Euro, der Zweitpreis 33,33 Euro.",
    "Ein Bieter mehr statt Mindestgebot": "Ein einziger Bieter mit dem besten Festpreis bringt 25,00 Euro, zwei Bieter ohne Mindestgebot 33,33 Euro (Bulow–Klemperer im einfachsten Fall).",
    "Unregulär (Bulow–Klemperer versagt)": "Mischung aus vielen niedrigen und wenigen hohen Werten, ein Bieter: das beste Verfahren bringt 9,00 Euro, zwei Bieter ohne Mindestgebot nur 4,53 Euro.",
    "Asymmetrisch": "Der zweite Bieter hat Werte nur bis 50 Euro: der Erstpreis bringt 22,95 Euro statt 20,83 Euro im Zweitpreis und vergibt den Slot in 8,33 % der Auktionen falsch.",
    "Lernende": "Zwei lernende Bieter im Erstpreis (Hedge, 20.000 Runden): nach dem Start in Reiter 5 steht der gelernte Gebotsfaktor neben dem Gleichgewicht 0,5.",
}
