"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Kennzahlen und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import eza_evaluation as V
from eza_distributions import LABELS
from eza_scenario import Settings

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "α": "alpha", "θ": "theta", "λ": "lambda", "σ": "sigma", "π": "pi", "δ": "delta", "η": "eta", "Δ": "Delta", "γ": "gamma", "ρ": "rho",
            "φ": "phi", "≈": "~", "½": "1/2", "…": "...", "‚": ",", "‘": "'", "’": "'", "”": '"', "µ": "mu", "∈": "in", "⁻": "^-", "¹": "1", "₀": "0", "₁": "1", "₂": "2", "ₘ": "m", "ₛ": "s",
            "⁎": "*", "±": "+-", "ⁿ": "^n", "⋅": "*"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1 (Umlaute gehören dazu): Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(st: Settings, view: V.MainView, verdicts: list[tuple[str, str]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Einzelgut-Auktion: wer die Regeln setzt"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Verteilung der Werte: {LABELS[st.dist]}; {st.n} Spediteure; Mindestgebot {V.de(st.r, 0)} Euro; Eigenwert des Verkäufers {V.de(st.v0, 0)} Euro je Slot."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Kennzahlen der Hauptansicht (Euro je Slot)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in (f"Erlös Zweitpreis: {V.de(view.spa, 2)}; Erlös Erstpreis: {V.de(view.fpa, 2)}; Verkaufswahrscheinlichkeit {V.pct(view.p_sale)}.",
                 f"Nutzen des Verkäufers: {V.de(view.utility, 2)} (ohne Mindestgebot über dem Eigenwert: {V.de(view.utility_plain, 2)}).",
                 f"Wohlfahrtsverlust bei diesem Mindestgebot: {V.de(view.loss, 2)} von {V.de(view.efficient, 2)}."):
        pdf.multi_cell(0, 5, _latin(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Verteilungen der Werte, Eigennutz, Unabhängigkeit und die Selbstbindung des Verkäufers sind Modellannahmen, keine Aussagen über echte Spediteure. "
                                "Alle Zahlen sind eigene Rechnung, gestützt durch Handrechnung, zweite Rechenwege (Integration, Optimierer, Beste-Antwort-Gitter) und Simulation; "
                                "die Lernläufe sind ein Befund über die gewählten Lernregeln."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
