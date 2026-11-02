"""Einzelgut-Auktion: wer die Regeln setzt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück der Linie „Regeln setzen, verhandeln, kontrollieren“. Ein Rampen- oder Laderaum-Slot geht an einen von n Spediteuren mit privaten Werten. Die Demo rechnet die Gleichgewichtsgebote von
Erstpreis, Zweitpreis und All-pay, die Erlösäquivalenz und ihre Voraussetzungen, das beste Mindestgebot, „ein Bieter mehr gegen das beste Festangebot“ (Bulow–Klemperer), asymmetrische Bieter
und lernende Bieter (gelb: dort landet mit Standardrate kein Verfahren im Gleichgewicht).

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import eza_constants as C
import eza_evaluation as V
import eza_experiments as XP
import eza_learning as LE
import eza_texts as T
import eza_visualization as Z
from eza_distributions import KINDS, LABELS
from eza_layout import dunkles_design, kopf_css
from eza_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    settings_from_state,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Einzelgut-Auktion – Sebastian Hanisch", layout="wide")


# --- Zwischenspeicher (Schlüssel = alle Eingaben inklusive Zufalls-Seed) --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _tables():
    return XP.load_tables()


@st.cache_data(show_spinner=False)
def _lernen():
    return XP.load_lernen()


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return V.analyse(settings)


@st.cache_data(show_spinner=False)
def _equivalence(settings):
    return V.equivalence_view(settings)


@st.cache_data(show_spinner=False)
def _risk(settings):
    return V.risk_view(settings)


@st.cache_data(show_spinner=False)
def _reserve(settings):
    return V.reserve_view(settings)


@st.cache_data(show_spinner=False)
def _bulow(settings):
    return V.bulow_view(settings)


@st.cache_data(show_spinner=False)
def _asym(settings):
    return V.asym_view(settings)


@st.cache_data(show_spinner=False)
def _raster_facts():
    from fractions import Fraction

    coarse = LE.class_fixed_points()
    fine_grid = [Fraction(k, 200) for k in range(201)]
    return {"coarse": [float(x) for x in coarse], "fine_best_on_045": float(LE.class_best_reply(Fraction(9, 20), fine_grid)),
            "fine_best_on_050": float(LE.class_best_reply(Fraction(1, 2), fine_grid))}


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in eza_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔨 Einzelgut-Auktion: wer die Regeln setzt")
st.markdown(T.INTRO)

with st.expander("So funktioniert eine Einzelgut-Auktion", expanded=True):
    st.markdown(T.EXPLAINER)

with st.expander("Erklärtabelle: was die Regler tun"):
    st.markdown(T.CONTROLS_TABLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🔨 Auktion")
    st.selectbox("Verteilung der Werte", KINDS, key="dist_select", format_func=LABELS.get,
                 help="Wie die Werte der Spediteure für den Slot streuen (Euro je Slot). Gilt für die Hauptansicht und die Reiter 1 bis 3.")
    st.slider("Zahl der Spediteure n", *bounds("n_slider"), key="n_slider", step=1, help="Wie viele Bieter es gibt. Mit n = 1 gibt es keinen Wettbewerb.")
    st.slider("Mindestgebot r (Euro je Slot)", *bounds("r_slider"), key="r_slider", step=1, help="Unter r wird nicht verkauft; wer verkauft, zahlt mindestens r.")
    st.slider("Eigenwert des Verkäufers v₀ (Euro je Slot)", *bounds("v0_slider"), key="v0_slider", step=1,
              help="Was der Slot dem Verkäufer bringt, wenn er unverkauft bleibt (andere Nutzung).")
    st.header("🎲 Risiko (Reiter 1)")
    st.slider("Risikoexponent ρ", *bounds("rho_slider"), key="rho_slider", step=0.05,
              help="Nutzen x^ρ der Spediteure; 1 ist risikoneutral. Der Risiko-Block rechnet immer mit Gleichverteilung auf 0 bis 100 Euro.")
    st.header("⚖️ Asymmetrie (Reiter 4)")
    st.slider("Schwäche w des zweiten Bieters", *bounds("w_slider"), key="w_slider", step=0.05,
              help="Der zweite Spediteur hat Werte gleichverteilt bis 100·w Euro, der erste bis 100 Euro. Reiter 4 rechnet immer mit zwei Bietern.")
    st.header("🧪 Lernende Bieter (Reiter 5)")
    st.selectbox("Lernregel", LE.RULES, key="rule_select", help="Hedge sieht, was jeder Gebotsfaktor gebracht hätte; Exp3 und Q-Lernen sehen nur die eigene Auszahlung.")
    st.selectbox("Format", LE.FORMATS, key="fmt_select", help="Erstpreis: Gleichgewicht α = (n − 1)/n; Zweitpreis: α = 1.")
    st.selectbox("Zahl der lernenden Bieter", C.LEARN_NS, key="nl_select", help="Werte gleichverteilt auf 0 bis 1 (= 100 Euro).")
    st.slider("Runden", *bounds("rounds_slider"), key="rounds_slider", step=1000, help="Wie viele Auktionen die Bieter lernen; ein Lauf dauert einige Sekunden (Knopf in Reiter 5).")
    st.select_slider("Lernrate (Vielfaches der Standardrate)", options=C.RATE_FACTORS, key="rate_select", format_func=lambda x: f"{V.de(x, 2)}×",
                     help="Vielfaches der Standardrate der Regel (Hedge: √(8 ln 21 / T); Exp3: √(K ln K / ((e − 1) T)); Q-Lernen: 0,05).")
    st.header("🎲 Zufall")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt die kurzen Simulationen der Reiter und den Lernlauf fest.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

settings = settings_from_state()
sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})

verdicts = []      # Verdikte für den PDF-Export
tables = _tables()
lernen = _lernen()
if tables is None or lernen is None:
    st.warning("Die vorgerechneten Messreihen (data/tables.json, data/lernen.json) fehlen oder gehören zu einem anderen Rechencode: `python tools/make_tables.py` und `python tools/make_lernen.py` ausführen. "
               "Die Experimente zeigen dann keine Reihen.")

view = _analysis(settings)

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 🎯 Ergebnis: Was bringt der Slot?")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Nutzen des Verkäufers", V.de(view.utility, 2), help="Erlös plus Eigenwert mal Wahrscheinlichkeit, dass der Slot unverkauft bleibt (Euro je Slot).")
m2.metric("Erlös Zweitpreis", V.de(view.spa, 2), help="Erwarteter Preis im Zweitpreis bei Mindestgebot r, gerechnet über die Ordnungsstatistik des zweithöchsten Werts.")
m3.metric("Erlös Erstpreis", V.de(view.fpa, 2), help="Erwartetes Höchstgebot im Erstpreis bei Mindestgebot r (Gleichgewichtsgebote).")
m4.metric("Verkauft mit", V.pct(view.p_sale), help="Wahrscheinlichkeit, dass mindestens ein Wert das Mindestgebot erreicht.")
best_r = view.r_star if view.r_star is not None else view.r_grid
m5.metric("Bestes Mindestgebot", V.de(best_r, 1), help="Reserve aus φ(r) = Eigenwert bei regulärer Verteilung, sonst das Maximum der Nutzenkurve auf dem Gitter.")
st.plotly_chart(Z.build_main(view), width="stretch", key="main_chart")
st.caption("Rot: Gebot im Erstpreis über dem eigenen Wert (der Abstand zur gepunkteten Linie ist das Bid Shading); gepunktet: Zweitpreis, dort ist das Gebot der Wert. Unter dem Mindestgebot wird nicht geboten.")
st.markdown("#### ❓ Kernfrage: Was ändert sich, wenn der Verkäufer die Regeln setzt?")
v_main = V.verdict_main(view)
show(v_main)
verdicts.append(v_main)

# --- Reiter --------------------------------------------------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 Erstpreis gegen Zweitpreis", "2 Mindestgebot", "3 Ein Bieter mehr gegen das beste Festangebot", "4 Asymmetrie", "5 🧪 Lernende Bieter"])

# --- Reiter 1 ------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Dieselben Erlöse, verschiedene Gebote")
    st.caption("**Erwartung:** Bei gleich starken, risikoneutralen Bietern mit unabhängigen Werten und ohne Mindestgebot bringen Erstpreis, Zweitpreis und All-pay im Mittel denselben Erlös; "
               "im Zweitpreis bietet jeder seinen Wert, im Erstpreis weniger (Bid Shading), im All-pay noch weniger, weil jeder zahlt. Live bei den Einstellungen der Seitenleiste (ohne Mindestgebot), "
               "Simulation: 4 Blöcke zu je 50.000 Auktionen.")
    eq = _equivalence(settings)
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(Z.build_bids(eq), width="stretch", key="bids_chart")
        st.caption("Gebot über dem eigenen Wert. Gepunktet: Zweitpreis (Gebot = Wert). Je mehr Konkurrenten, desto näher rückt das Erstpreis-Gebot an den Wert.")
    with col_b:
        st.plotly_chart(Z.build_revenue_bars(eq), width="stretch", key="revenue_chart")
        st.caption("Balken: Erlös aus dem Integral; Rauten: Simulation mit Doppel-Standardfehler.")
    v_eq = V.verdict_equivalence(eq)
    show(v_eq)
    verdicts.append(v_eq)
    if tables:
        st.markdown("**Vorgerechnet: Gleichverteilung auf 0 bis 100 Euro, n = 2 bis 10** (Erlös in Euro je Slot)")
        table(V.e1_table(tables))
        st.caption("Gebot bei Wert 50: Erstpreis. Zweitpreis auf zwei Wegen (A: virtuelle Werte, B: Ordnungsstatistik), Erstpreis über das Gleichgewichtsgebot, All-pay über seine eigene Gebotsfunktion.")
        table(V.e1_sim_table(tables))
        st.caption(f"Simulation: {tables['blocks']} Blöcke zu je {tables['auctions_per_block']:,} Auktionen".replace(",", ".") + ", ± ist der Standardfehler der Blockwerte. Beste-Antwort-Lücke: größter Gewinn, "
                   "den ein Bieter mit Wert 20, 50, 80 oder 100 Euro durch Abweichen vom Gleichgewichtsgebot hätte, bei 6.001 geprüften Geboten.")
        v1 = V.verdict_e1_tables(tables)
        show(v1)
        verdicts.append(v1)
        st.markdown("**Andere Verteilungen** (die Äquivalenz hängt nicht an der Gleichverteilung)")
        table(V.e1_other_table(tables))
        st.markdown("**Negativkontrolle des Beste-Antwort-Prüfers:** bieten die Gegner fälschlich w²/100 statt w/2, muss das Gleichgewichtsgebot v/2 eine Lücke haben")
        table(V.e1_control_table(tables))
    st.markdown("---")
    st.subheader("🔬 Experiment E2: Risikoscheu bricht die Äquivalenz")
    st.caption("**Erwartung:** Wer das Risiko scheut, bietet im Erstpreis mehr (sicher gewinnen statt viel gewinnen), der Zweitpreis bleibt bei ehrlichen Geboten; der Erstpreis bringt dann mehr, und der Abstand "
               "schrumpft mit der Zahl der Bieter. Nutzen x^ρ, Werte immer gleichverteilt auf 0 bis 100 Euro.")
    rk = _risk(settings)
    st.plotly_chart(Z.build_risk(settings.n, settings.rho), width="stretch", key="risk_chart")
    st.caption("Verhältnis Erlös Erstpreis zu Erlös Zweitpreis über dem Risikoexponenten, für n = 2, 5, 10 und das gewählte n; Raute: gewählte Einstellung.")
    v_rk = V.verdict_risk(rk)
    show(v_rk)
    verdicts.append(v_rk)
    if tables:
        table(V.e2_table(tables))
        v2 = V.verdict_e2_tables(tables)
        show(v2)
        verdicts.append(v2)

# --- Reiter 2 ------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E3: Was ist ein Mindestgebot wert, und was kostet es?")
    st.caption("**Erwartung:** Ein Mindestgebot über dem Eigenwert hebt den Preis, senkt aber die Verkaufswahrscheinlichkeit; das beste r* löst φ(r*) = Eigenwert und hängt bei Gleichverteilung nicht von n ab; "
               "sein Gewinn schrumpft mit n, und sein Preis ist Wohlfahrt (der Slot bleibt bei kleinen Werten unverkauft). Erst- und Zweitpreis bringen mit Mindestgebot dasselbe.")
    rs = _reserve(settings)
    st.plotly_chart(Z.build_reserve(rs), width="stretch", key="reserve_chart")
    st.caption("Nutzen des Verkäufers über dem Mindestgebot (Eigenwert, Verteilung und n der Seitenleiste). Die Kurve ist flach um das Maximum: ein ungefähres r* verliert wenig.")
    v_rs = V.verdict_reserve(rs)
    show(v_rs)
    verdicts.append(v_rs)
    st.markdown(f"**Beim gewählten Mindestgebot {V.de(settings.r, 0)} Euro:** Erstpreis {V.de(rs.first_at_reserve, 3)} Euro, Zweitpreis {V.de(rs.second_at_reserve, 3)} Euro; im Erstpreis bietet der Bieter mit dem "
                f"Wert des Mindestgebots genau {V.de(rs.bid_at_reserve, 3)} Euro; größte Beste-Antwort-Lücke {V.sci(rs.gap_first)} Euro.")
    if tables:
        st.markdown("**Vorgerechnet: Gleichverteilung, Eigenwert 0, 20, 40 Euro, n = 1 bis 10** (Nutzen in Euro je Slot)")
        table(V.e3_table(tables))
        st.caption("r*: Formel (100 + v₀)/2, Nullstelle von φ(r) = v₀ (Bisektion) und Gittersuche (Weg B, Schritt 0,005 Euro). Gewinn: gegenüber einem Mindestgebot gleich dem Eigenwert. "
                   "Wohlfahrtsverlust in Prozent des Werts E[max(Höchstwert, Eigenwert)].")
        v3 = V.verdict_e3_tables(tables)
        show(v3)
        verdicts.append(v3)
        st.markdown("**Andere Verteilungen**")
        table(V.e3_other_table(tables))
        st.markdown("**Simulation gegen die beiden Rechenwege** (Gleichverteilung, Zweitpreis, Eigenwert 20 Euro)")
        table(V.e3_sim_table(tables))
        st.markdown("**Erstpreis mit Mindestgebot gleich Zweitpreis mit Mindestgebot** (Gleichverteilung)")
        table(V.e3_fpa_table(tables))

# --- Reiter 3 ------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E4: Ein Bieter mehr gegen das beste Festangebot")
    st.caption("**Erwartung (Bulow–Klemperer):** Eine einfache Auktion mit einem Bieter mehr bringt dem Verkäufer mindestens so viel wie das beste Verfahren mit n Bietern, wenn die Werte unabhängig, "
               "die Bieter risikoneutral, ihre Werte mindestens der Eigenwert des Verkäufers und die Verteilung regulär sind. „Das beste Verfahren“ meint hier die Auktion mit Selbstbindung "
               "(Zweitpreis mit bestem Mindestgebot), nicht abwechselnde Angebote wie im Verhandlungsstück. Jede Voraussetzung wird unten verletzt.")
    bk = _bulow(settings)
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(Z.build_bulow(bk), width="stretch", key="bulow_chart")
        st.caption("Nutzen des Verkäufers: bestes Verfahren mit n Bietern (orange) gegen einfache Auktion mit n + 1 Bietern (grün).")
    with col_d:
        st.plotly_chart(Z.build_threshold(bk), width="stretch", key="threshold_chart")
        st.caption("Teilnahmewahrscheinlichkeit, ab der der zusätzliche Bieter genügt (logarithmische Achse). Rot: auch bei sicherer Teilnahme reicht ein Bieter nicht.")
    v_bk = V.verdict_bulow(settings.dist, settings.v0, bk.row)
    show(v_bk)
    verdicts.append(v_bk)
    v_bk2 = V.verdict_bulow_flat(settings.dist, [r for r in bk.rows if r["n"] <= 10], settings.v0)
    show(v_bk2)
    verdicts.append(v_bk2)
    if tables:
        series = [(k, 0.0) for k in ("gleich", "exp", "beta")] + [(k, C.E4_V0) for k in ("gleich", "exp", "beta")] + [("mix", 0.0)]
        names = [f"{V.DIST_SHORT[k]}, Eigenwert {V.de(v, 0)} Euro" for k, v in series]
        choice = st.selectbox("Vorgerechnete Reihe", names, index=0, key="e4_series")
        kind, v0 = series[names.index(choice)]
        table(V.e4_table(tables, kind, v0))
        st.caption("Bestes Verfahren: Hülle der Erlöskurve im Quantilenraum (bei regulärer Verteilung gegen die Reserve aus φ(r) = v₀ geprüft). Einfach: Zweitpreis mit Mindestgebot gleich dem Eigenwert. "
                   "Zusätzliche Bieter: kleinste Zahl k, mit der die einfache Auktion das beste Verfahren erreicht. Schwelle p*: Teilnahmewahrscheinlichkeit des einen zusätzlichen Bieters.")
        v4 = V.verdict_e4_tables(tables)
        show(v4)
        verdicts.append(v4)

# --- Reiter 4 ------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E5: Ungleich starke Bieter")
    st.caption("**Erwartung:** Hat der zweite Bieter niedrigere Werte (bis 100·w Euro), bieten beide im Erstpreis verschieden; der Erstpreis bringt dann mehr als der Zweitpreis und vergibt den Slot manchmal nicht an den "
               "höheren Wert (der Zweitpreis nie). Zwei Bieter, Werte gleichverteilt, Gleichgewicht in geschlossener Form (hergeleitet und gemessen).")
    asym = _asym(settings)
    st.plotly_chart(Z.build_asym(asym), width="stretch", key="asym_chart")
    st.caption("Gebot über dem eigenen Wert. Der schwache Bieter bietet bei gleichem Wert mehr als der starke; höchstes Gebot beider: "
               f"{V.de(asym.top_bid, 2)} Euro.")
    v_as = V.verdict_asym(asym)
    show(v_as)
    verdicts.append(v_as)
    if tables:
        st.markdown("**Vorgerechnet: Erlöse und Wohlfahrt** (Euro je Slot)")
        table(V.e5_table(tables))
        st.caption("Myerson: Mindestgebote je Bieter (virtuelle Werte 2v − 100 und 2v − 100·w, Zuschlag an den größten positiven virtuellen Wert, Schwellenzahlung). Gilt für diese Verteilungsfamilie, nicht allgemein.")
        table(V.e5_check_table(tables))
        st.caption(f"Simulation: {tables['blocks']} Blöcke zu je {tables['asym_auctions_per_block']:,} Auktionen".replace(",", ".") + ". Beste-Antwort-Lücke: größter Gewinn durch Abweichen, beide Bieter, vier Werte je Bieter.")
        v5 = V.verdict_e5_tables(tables)
        show(v5)
        verdicts.append(v5)

# --- Reiter 5 ------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.warning("🧪 **Gelb markiert: Befund, keine Prüfung der Gleichgewichtsgebote.** Dieser Reiter zeigt, was lernende Bieter tun, und der Befund ist ehrlich negativ: mit Standardrate landet kein Verfahren im Gleichgewicht, "
               "die Richtung hängt von der Lernregel ab, und das Raster der Gebotsfaktoren erzeugt eigene Gleichgewichte.")
    st.subheader("🔬 Experiment E6: Finden lernende Bieter die Gleichgewichtsgebote?")
    st.caption("**Erwartung:** Lernen mit Rückmeldung sollte im Erstpreis auf den Gebotsfaktor (n − 1)/n und im Zweitpreis auf 1 zulaufen, und der Erlös beider Formate gleich bleiben. "
               "Werte gleichverteilt auf 0 bis 1 (= 100 Euro), Gebot = α · Wert, α auf einem Raster von 21 Werten (Q-Lernen: 11 Werte, 10 Wertklassen).")
    st.markdown(f"**Ein Lernlauf live** ({settings.rule}, {settings.fmt}, {settings.n_learn} Bieter, {V.de(settings.rounds, 0)} Runden, Rate {V.de(settings.rate, 2)}×; Dauer einige Sekunden)")
    if st.button("▶️ Lernlauf starten", key="learn_run"):
        t0 = time.time()
        with st.spinner("Der Lernlauf läuft ..."):
            res = V.run_learning(settings)
        res["rounds"] = settings.rounds
        res["seconds"] = time.time() - t0
        st.session_state["learn_result"] = {"key": (settings.rule, settings.fmt, settings.n_learn, settings.rounds, settings.rate, settings.seed), "res": res}
    stored = st.session_state.get("learn_result")
    current_key = (settings.rule, settings.fmt, settings.n_learn, settings.rounds, settings.rate, settings.seed)
    if stored and stored["key"] == current_key:
        res = stored["res"]
        st.plotly_chart(Z.build_learning(res), width="stretch", key="learn_chart")
        st.caption(f"Verlauf des mittleren Gebotsfaktors (Mittel der Bieter) über dem Lauf; Laufzeit {V.de(res['seconds'], 1)} s.")
        v_l = V.verdict_learning(res)
        show(v_l)
        verdicts.append(v_l)
    elif stored:
        st.info("Die Einstellungen haben sich seit dem letzten Lauf geändert: Lernlauf neu starten.")
    else:
        st.info("Noch kein Lauf: ▶️ startet einen Lernlauf mit den Einstellungen der Seitenleiste.")
    if lernen:
        st.markdown("**Vorgerechnet: letzter Iterierter nach langen Läufen** (Hedge 20.000, Exp3 100.000, Q-Lernen 300.000 Runden; 4 Läufe je Zelle)")
        table(V.e6_table(lernen))
        st.plotly_chart(Z.build_learning_cells([c for c in lernen["cells"] if c["eta"] is None and c["rounds"] == C.LEARN_T[c["rule"]]]), width="stretch", key="learn_cells_chart")
        st.caption("Balken: letzter Gebotsfaktor mit Doppel-Standardfehler über 4 Läufe; Strich: Gleichgewicht. Erlös in Einheiten von 100 Euro; Gleichgewichtserlös (n − 1)/(n + 1).")
        v6 = V.verdict_e6_tables(lernen)
        show(v6)
        verdicts.append(v6)
        st.markdown("**Horizont und Rate (Hedge, zwei Bieter)**")
        table(V.e6_horizon_table(lernen))
        fixed = V.learn_cell(lernen, "Hedge", 2, "Erstpreis", 20000, C.LEARN_FIXED_ETA)
        hits = sum(1 for x in fixed["alpha_last_blocks"] if abs(x - 0.45) < 5e-4)
        raster = _raster_facts()
        st.markdown(f"**Rasterfalle (exakt gerechnet):** im Klassenspiel b = α · v mit Raster-Schritt 0,05 sind die Faktoren {' und '.join(V.de(x, 2) for x in raster['coarse'])} beide Beste Antwort auf sich selbst, "
                    f"obwohl nur 0,5 das Gleichgewicht des Auktionsspiels ist. Mit einem feinen Raster (Schritt 0,005) wäre die Beste Antwort auf 0,45 der Faktor {V.de(raster['fine_best_on_045'], 3)}. "
                    f"Hedge mit fester Rate 1,0 landet nach 20.000 Runden in {hits} von {len(fixed['alpha_last_blocks'])} Läufen bei 0,450 und in "
                    f"{len(fixed['alpha_last_blocks']) - hits} bei einem anderen Wert ({', '.join(V.de(x, 3) for x in fixed['alpha_last_blocks'] if abs(x - 0.45) >= 5e-4)}): "
                    "das Raster erzeugt Gleichgewichte, die es im Auktionsspiel nicht gibt.")
        st.markdown("**Einordnung:** Die Abweichung ist ein Befund über die gewählten Lernregeln, Raten und das Raster, kein Befund über echte Spediteure. Banchio und Skrzypacz (2022) fanden laut Abstract, dass "
                    "Q-Lernen im Erstpreis ohne weitere Rückmeldung zu stillschweigend abgestimmten, niedrigen Geboten (Kollusion) führt, im Zweitpreis nicht; das ist hier **nicht reproduziert** (anderes Modell: Werte pro Runde, 10 Klassen, kein Zustand; Q-Lernen liegt hier im Erstpreis über dem Gleichgewicht).")

# --- Verwandte Demos, Grenzen, Formeln -----------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    "- [`verhandlung-demo`](https://github.com/sebastian-hanisch/verhandlung-demo): volle Information, Angebot und Gegenangebot in Runden; hier steht ein Zug mit privatem Wert dagegen.\n"
    "- `auction-demo`: kombinatorische Beschaffungsauktion mit Fertigstellungszeiten als Geboten; keine Verteilung der Werte, kein Gleichgewicht.\n"
    "- `myerson-satterthwaite-demo`: zwei private Seiten (Käufer und Verkäufer) für denselben Slot, Mechanismen als lineares Programm.\n"
    "- `regretnet-demo`: gelernte Mechanismen für die Kran-Beschaffung."
)

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.MATH_MARKDOWN)

st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from eza_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(settings, view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="einzelgut-auktion.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
