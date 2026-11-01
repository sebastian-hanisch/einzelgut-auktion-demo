"""Feste Texte der App, die auch in der README stehen (ein Test vergleicht beide Fassungen, DEMO-PLAYBOOK §5). Messzahlen kommen aus der Rechnung (eza_evaluation);
die wenigen festen Zahlen in `ASSUMPTIONS_MARKDOWN` sind in tests/test_claims.py gegen die vorgerechneten Reihen gepinnt."""

INTRO = """
**Der Verkäufer legt die Regeln fest:** Ein **Rampen- oder Laderaum-Slot** geht an einen von **n Spediteuren**, und jeder kennt nur seinen eigenen **Wert** für den Slot (Euro je Slot). Anders als im Verhandlungsstück
([`verhandlung-demo`](https://github.com/sebastian-hanisch/verhandlung-demo): volle Information, Angebot und Gegenangebot in Runden) gibt es hier **einen Zug**: der Verkäufer wählt **Format** (Erstpreis, Zweitpreis, All-pay)
und **Mindestgebot**, die Spediteure antworten im **Gleichgewicht** der Gebote. Gezeigt wird, wann die Formate **dasselbe** bringen (Erlösäquivalenz) und wann nicht, was ein **Mindestgebot** wert ist und was es kostet,
wann **ein Bieter mehr** das beste Festangebot schlägt (Bulow–Klemperer), was **ungleich starke** Bieter ändern und was **lernende** Bieter tun (gelb markiert: dort landet kein Verfahren im Gleichgewicht).

Abgrenzung: `auction-demo` vergibt mehrere Kran-Aufträge als Bündel (kombinatorisch, ohne Verteilung der Werte, ohne Gleichgewicht); `myerson-satterthwaite-demo` hat **zwei** private Seiten (Käufer und Verkäufer) und rechnet
Mechanismen als lineares Programm; `regretnet-demo` lernt Mechanismen. Hier ist **eine** Seite privat, das Gut ein einzelner Slot, und das Gleichgewicht wird für beliebige Verteilungen aus einem Integral berechnet.
**Verteilungen der Werte, Eigennutz, Unabhängigkeit der Werte und die Selbstbindung des Verkäufers sind Modellannahmen**, keine Aussagen über echte Spediteure; die Zahlen sind erfunden.

Weiter unten: fünf Reiter mit sechs **🔬 Experimenten**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

EXPLAINER = r"""
1. **Der Verkäufer legt vorab fest und bindet sich daran:** Format und Mindestgebot $r$. Sein Eigenwert $v_0$ ist das, was ihm der Slot bringt, wenn er unverkauft bleibt.
2. **Jeder Spediteur $i$ kennt nur seinen Wert $v_i$**, die anderen kennen nur die Verteilung $F$ (unabhängig gezogen). Im **Zweitpreis** zahlt der Höchstbietende das zweithöchste Gebot (mindestens $r$): ehrlich zu bieten ist dann die beste Antwort.
3. **Im Erstpreis** zahlt der Höchstbietende sein eigenes Gebot und bietet deshalb **weniger als seinen Wert** („Bid Shading“). Wie viel weniger, folgt aus der Frage, wie viele Konkurrenten er schlagen muss.
4. **Im All-pay** zahlt jeder sein Gebot, auch wer verliert; gewinnt der Höchste, ist es dem Verkäufer trotzdem recht.
5. **Erlösäquivalenz.** Sind die Bieter gleich stark, risikoneutral und ihre Werte unabhängig, bringen alle drei Regeln im Mittel denselben Erlös. Jede Voraussetzung ist ein Reiter, in dem sie verletzt wird.
6. **Mindestgebot.** Es senkt die Chance, den Slot zu verkaufen, hebt aber den Preis: das beste $r^*$ löst $\varphi(r^*) = v_0$ mit dem virtuellen Wert $\varphi(v) = v - \frac{1 - F(v)}{f(v)}$.
"""

CONTROLS_TABLE = """
| Regler | Was er tut | Wirkung auf die Frage |
|---|---|---|
| **Verteilung** | Wie die Werte der Spediteure streuen (Euro je Slot): gleichverteilt, exponentiell, Beta(2, 2) oder eine Mischung (viele niedrige, wenige hohe Werte). | Bestimmt Gebote, Erlöse und Reserve; bei der Mischung steigt der virtuelle Wert nicht (unregulär). Gilt für Hauptansicht und die Reiter 1 bis 3. |
| **Zahl der Spediteure n** | Wie viele Bieter es gibt (1 bis 20). | Mehr Konkurrenz zieht die Gebote an den Wert und den Erlös an den zweithöchsten Wert. |
| **Mindestgebot r** | Unter r wird nicht verkauft; wer verkauft, zahlt mindestens r. | Hebt den Preis, senkt die Verkaufswahrscheinlichkeit. |
| **Eigenwert v0 des Verkäufers** | Was der Slot ihm bringt, wenn er unverkauft bleibt (Euro). | Verschiebt das beste Mindestgebot nach oben. |
| **Risikoexponent ρ** | Nutzen x^ρ der Spediteure (1 = risikoneutral); wirkt im Risiko-Block von Reiter 1 (immer Gleichverteilung). | Bricht die Erlösäquivalenz: Erstpreis über Zweitpreis. |
| **Schwäche w des zweiten Bieters** | Der zweite Spediteur hat Werte nur bis 100·w Euro (immer Gleichverteilung); wirkt in Reiter 4. | Bricht die Erlösäquivalenz und macht den Erstpreis ineffizient. |
| **Lernregel, Format, Bieter, Runden, Lernrate** | Reiter 5: wie viele Runden zwei oder vier Bieter lernen (Werte gleichverteilt), mit welcher Regel und Rate (Vielfaches der Standardrate). | Zeigt, dass gelernte Gebote nicht im Gleichgewicht liegen. |
| **Zufalls-Seed** | Legt die Simulationen der Reiter und den Lernlauf fest. | Gleicher Seed, gleiches Ergebnis. |
"""

ASSUMPTIONS_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wo gemessen |
|---|---|---|
| **Risikoneutral** (Nutzen = Euro) | Bei Risikoscheu (Exponent 0,5, zwei Bieter) bringt der Erstpreis 44,44 statt 33,33 Euro im Zweitpreis, Verhältnis 1,3333; mit zehn Bietern nur noch 1,0526. | E2 |
| **Gleich starke Bieter** | Hat der zweite Bieter Werte nur bis 50 Euro (w = 0,5), bringt der Erstpreis 22,95 statt 20,83 Euro (Verhältnis 1,1016) und vergibt den Slot in 8,33 % der Auktionen nicht an den höheren Wert. | E5 |
| **Reguläre Verteilung** (der virtuelle Wert steigt) | Bei der Mischung gilt „ein Bieter mehr schlägt das beste Verfahren“ in 0 von 15 Zeilen; bei einem Bieter bringt das beste Verfahren 9,00 Euro, zwei Bieter ohne Mindestgebot 4,53 Euro. | E4 |
| **Bieterwert ≥ Eigenwert des Verkäufers** („serious bidder“) | Exponentiell verteilte Werte mit Eigenwert 20 Euro: bei einem Bieter 31,23 Euro (zwei Bieter) gegen 32,33 Euro (bestes Verfahren), die Aussage gilt in 9 von 10 Zeilen. | E4 |
| **Unabhängige Werte, ein Gut, keine Absprache, keine Teilnahmekosten** | Nicht gerechnet. Mehrere Plätze (Positionsauktion) rechnet diese Demo nicht: sie hat einen Slot. | – |
| **Der Verkäufer bindet sich** an Format und Mindestgebot | Nicht gerechnet: ein Verkäufer, der nach einem Fehlschlag nachverhandelt, ändert die Anreize der Bieter. | – |
| **Die Bieter spielen das Gleichgewicht** | Lernende Bieter landen nicht dort: Hedge mit zwei Bietern im Erstpreis kommt nach 20.000 Runden auf den Gebotsfaktor 0,456 statt 0,5 (4 Läufe). Das ist ein Befund über die gewählten Lernregeln und das Raster, nicht über echte Spediteure. | E6 |
| **Literatur nur teilweise im Volltext gelesen** | Bulow und Klemperer (1996) wurde im Volltext gelesen (laut Einzelplan); Myerson (1981), Vickrey (1961), Riley und Samuelson (1981), Griesmer, Levitan und Shubik (1967) und Banchio und Skrzypacz (2022) nicht. Die Demo rechnet das beschriebene Modell nach und behauptet keine ihrer Aussagen. | – |
"""

MATH_MARKDOWN = r"""
**Lesehilfe.** Gegeben sind $n$ Bieter mit unabhängigen Werten $v_i \sim F$ (Dichte $f$), ein Mindestgebot $r$ und der Eigenwert $v_0$ des Verkäufers; gesucht sind die Gleichgewichtsgebote, die Erlöse und das beste $r$.

| Symbol | Bedeutung |
|---|---|
| $F, f$ | Verteilungsfunktion und Dichte der Werte (Euro je Slot) |
| $n$ | Zahl der Bieter |
| $r$, $v_0$ | Mindestgebot und Eigenwert des Verkäufers |
| $\varphi(v)$ | virtueller Wert $v - \frac{1 - F(v)}{f(v)}$ |
| $b(v)$ | Gleichgewichtsgebot eines Bieters mit Wert $v$ |
| $R(r)$, $U(r)$ | Erlös und Nutzen des Verkäufers bei Mindestgebot $r$ |
| $\rho$ | Risikoexponent, Nutzen $x^{\rho}$ |
| $w$ | Schwäche des zweiten Bieters (Werte bis $100\,w$ Euro) |
| $R(q)$, $\overline{MR}$ | Erlöskurve im Quantilenraum und die Steigung ihrer oberen konkaven Hülle |

**Zweitpreis.** Ehrlich zu bieten ist dominant, der Preis ist $\max(r, v_{(2)})$. Der Erlös lässt sich auf zwei Wegen rechnen, über den erwarteten virtuellen Wert des Höchsten

$$R(r) = n \int_r^{\infty} \varphi(v)\, F(v)^{n-1} f(v)\, dv$$

und über die Ordnungsstatistik des zweithöchsten Werts

$$R(r) = r \cdot n F(r)^{n-1}\bigl(1 - F(r)\bigr) + \int_r^{\infty} v\, n(n-1) F(v)^{n-2}\bigl(1 - F(v)\bigr) f(v)\, dv .$$

**Erstpreis.** Wer den Wert $v \ge r$ hat, bietet im Gleichgewicht

$$b(v) = v - \int_r^{v} \left(\frac{F(x)}{F(v)}\right)^{n-1} dx ,$$

bei Gleichverteilung auf $[0, 100]$ ohne Mindestgebot also $b(v) = \frac{n-1}{n}\, v$. Das Gebot löst die Bedingung erster Ordnung $b'(v) = (n-1)\,\frac{f(v)}{F(v)}\,(v - b(v))$.

**All-pay** (ohne Mindestgebot). Jeder zahlt sein Gebot, die Bedingung erster Ordnung gibt

$$b(v) = \int_0^{v} x\, (n-1)\, F(x)^{n-2} f(x)\, dx .$$

**Erlösäquivalenz.** Der Bieter mit dem niedrigsten teilnehmenden Wert $r$ erwartet 0; dann ist die erwartete Zahlung eines Bieters mit Wert $v$ in allen drei Regeln gleich

$$m(v) = v\, F(v)^{n-1} - \int_r^{v} F(x)^{n-1}\, dx ,$$

und damit auch der Erlös. Die Demo rechnet jede Regel mit ihrer eigenen Gebotsfunktion und prüft sie gegen das Beste-Antwort-Gitter, die Gleichheit ist also gemessen, nicht eingesetzt.

**Risikoscheu** (Nutzen $x^{\rho}$, Gleichverteilung, Erstpreis). Die Bedingung erster Ordnung gibt den Gebotsfaktor

$$b(v) = \frac{n-1}{n-1+\rho}\, v ,$$

größer als $\frac{n-1}{n}$ für $\rho < 1$; der Zweitpreis bleibt bei ehrlichen Geboten.

**Mindestgebot.** Der Nutzen des Verkäufers ist

$$U(r) = R(r) + v_0\, F(r)^n ,$$

und bei regulärer Verteilung ($\varphi$ steigt) maximiert $r^*$ mit

$$\varphi(r^*) = v_0$$

den Nutzen. Bei $U[0, 100]$ ist $r^* = \frac{100 + v_0}{2}$, unabhängig von $n$. Mit Erstpreis und Mindestgebot ist der Erlös derselbe wie im Zweitpreis mit Mindestgebot.

**Ein Bieter mehr.** Im Quantilenraum $q = 1 - F(v)$ ist die Erlöskurve $R(q) = q\, F^{-1}(1-q)$; $\overline{MR}$ sei die Steigung ihrer oberen konkaven Hülle. Das beste Verfahren mit $n$ Bietern erreicht

$$U^{\mathrm{opt}}(n) = \mathbb{E}\bigl[\max(\overline{MR}(q_1),\, v_0)\bigr], \qquad q_1 \sim n(1-q)^{n-1},$$

die einfache Auktion mit $n+1$ Bietern (Zweitpreis, Mindestgebot $v_0$) ohne Eigenwert

$$R_{n+1} = (n+1)\, n \int_0^1 R(q)\, (1-q)^{n-1}\, dq .$$

Bulow und Klemperer zeigen $R_{n+1} \ge U^{\mathrm{opt}}(n)$ unter unabhängigen Signalen, Risikoneutralität, Bieterwert $\ge$ Eigenwert und Regularität.

**Asymmetrie** (Wert 1 aus $U[0, 100]$, Wert 2 aus $U[0, 100\,w]$, Einheiten von 100 Euro). Mit $c = \frac{w^2 - 1}{w^2}$ sind die Umkehrfunktionen der Gleichgewichtsgebote

$$x(b) = \frac{2b}{1 + c b^2}, \qquad y(b) = \frac{2b}{1 - c b^2}, \qquad \bar b = \frac{w}{1 + w},$$

mit $x'(y - b) = x$ und $y'(x - b) = y$ sowie $x(\bar b) = 1$, $y(\bar b) = w$ (hergeleitet und gemessen, hier nicht aus der Literatur übernommen).

**Lernen.** Gebote $b = \alpha\, v$ mit $\alpha$ auf einem Raster von 21 Werten. *Hedge* gewichtet $\alpha$ mit $\exp(-\eta\, L_\alpha)$ (volle Rückmeldung), *Exp3* beobachtet nur die eigene Auszahlung, *Q-Lernen* lernt je Wertklasse.

**Literatur.** Bulow und Klemperer, „Auctions versus Negotiations“, American Economic Review 86(1), 1996, S. 180–194: laut Einzelplan im Volltext gelesen. Myerson, „Optimal Auction Design“, Mathematics of Operations Research 6(1), 1981, S. 58–73,
Vickrey 1961, Riley und Samuelson 1981, Griesmer, Levitan und Shubik 1967 sowie Banchio und Skrzypacz 2022 (arXiv 2202.05947): **laut Auszug oder Titel, nicht im Volltext geprüft**.
"""
