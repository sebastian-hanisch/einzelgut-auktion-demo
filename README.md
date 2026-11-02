# Einzelgut-Auktion: wer die Regeln setzt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-einzelgut-auktion-demo.streamlit.app/)** -->

Stück der Linie **regeln-verhandeln** (Arbeitsname; Name, Bereich und Themenfeld werden erst bei der Einbindung in die Website festgelegt) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es folgt auf [`verhandlung-demo`](https://github.com/sebastian-hanisch/verhandlung-demo), in dem zwei Betriebe bei vollständiger Information einen Überschuss in Runden teilen; hier legt **der Verkäufer die Regeln vorab fest**, die Gegenseite hat private Werte, und es gibt **einen Zug** statt Angebot und Gegenangebot. Vehikel: ein **Rampen- oder Laderaum-Slot**, **n Spediteure** mit privaten Werten (Euro je Slot), ein Verkäufer mit Eigenwert für den unverkauften Slot. Verteilungen der Werte, Eigennutz, Unabhängigkeit der Werte und die Selbstbindung des Verkäufers sind **Modellannahmen**, keine Aussagen über echte Spediteure; die Zahlen sind erfunden.

Abgrenzung: [`auction-demo`](https://github.com/sebastian-hanisch/auction-demo) vergibt mehrere Kran-Aufträge als Bündel (kombinatorisch, Gebote sind Fertigstellungszeiten, Zahlungsregeln Pay-as-bid gegen VCG) und hat weder eine Verteilung der Werte noch ein Gleichgewicht noch einen Erlös. [`myerson-satterthwaite-demo`](https://github.com/sebastian-hanisch/myerson-satterthwaite-demo) hat für denselben Slot **zwei** private Seiten (Käufer und Verkäufer) und rechnet Mechanismen als lineares Programm; das Gleichgewicht einer Einzelgut-Auktion baut es ausdrücklich nicht nach. [`regretnet-demo`](https://github.com/sebastian-hanisch/regretnet-demo) lernt Mechanismen für die Kran-Beschaffung. [`noregret-demo`](https://github.com/sebastian-hanisch/noregret-demo) und [`algorithmische-kollusion-demo`](https://github.com/sebastian-hanisch/algorithmische-kollusion-demo) nutzen Hedge, EXP3 und Q-Lernen in anderen Spielen. Eine Zusammenlegung scheidet aus: eine Verteilungsrechnung mit Integralen und ein LP-Gerüst passen nicht als Reiter in diese Demos, und ein Update live stehender Demos wäre eine eigene Freigabe mit Redeploy.

## Kernfrage

*Was ändert sich, wenn der Verkäufer die Spielregeln festlegt, statt zu verhandeln, und wie viel ist eine Regel wert?* Die Antwort hat vier Teile. (1) **Das Format ist unter engen Voraussetzungen egal, der Wettbewerb nicht:** bei gleich starken, risikoneutralen Bietern mit unabhängigen Werten bringen Erstpreis, Zweitpreis und All-pay denselben Erlös (bei Gleichverteilung auf 0 bis 100 Euro 33,33 Euro bei zwei und 81,82 Euro bei zehn Bietern), aber verschiedene Gebote: im Erstpreis bietet der Spediteur mit Wert 50 Euro 25,00 Euro bei zwei und 45,00 Euro bei zehn Bietern. (2) **Die Äquivalenz bricht, sobald eine Voraussetzung fällt:** bei Risikoscheu (Exponent 0,5, zwei Bieter) bringt der Erstpreis 1,3333-fach den Zweitpreis, bei einem zweiten Bieter mit Werten nur bis 50 Euro 1,1016-fach, und der Erstpreis vergibt den Slot dann in 8,33 % der Auktionen nicht an den höheren Wert. (3) **Das beste Mindestgebot r\* = (100 + v₀)/2 hebt den Nutzen** (bei zwei Bietern ohne Eigenwert um 25,0 %), **kostet aber Wohlfahrt** (12,5 % des effizienten Werts), und **ein Bieter mehr** schlägt das beste Verfahren (bei einem Bieter ein Festangebot, sonst die Auktion mit bestem Mindestgebot; Bulow–Klemperer) in 30 von 30 Zeilen der regulären Reihen, aber in 0 von 15 Zeilen der Mischung (n bis 60) und nicht, wenn der Eigenwert des Verkäufers über manchen Bieterwerten liegt (9 von 10 Zeilen bei Exponentialverteilung mit Eigenwert 20 Euro). (4) **Lernende Bieter fanden die Gleichgewichtsgebote in der Messung nicht** (gelber Reiter): in 10 von 10 Zellen mit Standardrate (Raster 0,05) weicht der gelernte Gebotsfaktor ab; je nach Lernregel liegt mal der Erst-, mal der Zweitpreis vorn (bei Exp3 mit vier Bietern gleich). Hedge mit fester Rate 1,0 erreichte im Zweitpreis das Gleichgewicht.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (Vormessung vom 2026-11-01, Wegwerfskripte) wurde nicht blind übernommen; jede Zahl ist im Bau neu gemessen, mit SplitMix64-Strömen statt der numpy-Ströme der Vormessung. Abweichungen und neue Funde:

- **Das Muster p\* = (n + 2)/2ⁿ⁺¹ ist bewiesen, nicht mehr offen.** Bei U[0, 1] und Eigenwert 0 ist der Nutzen des besten Verfahrens (n − 1)/(n + 1) + 2⁻ⁿ/(n + 1), der Zweitpreis ohne Mindestgebot (n − 1)/(n + 1) und mit einem Bieter mehr n/(n + 2); der Quotient (2⁻ⁿ/(n + 1)) : (2/((n + 1)(n + 2))) ist (n + 2)/2ⁿ⁺¹. Die Suite rechnet es in Brüchen für n = 1 bis 12 nach und vergleicht mit der gemessenen Schwelle für n = 1 bis 10.
- **Der Rasterfall des Lernens ist exakt gerechnet.** Im Klassenspiel b = α · v auf dem Raster mit Schritt 0,05 sind α = 0,45 und α = 0,50 beide Beste Antwort auf sich selbst (Brüche, ein geschlossenes Integral; keine andere Zahl des Rasters). Der Plan nannte für das feine Raster (Schritt 0,005) als Beste Antwort auf 0,50 den Wert 0,495; in Brüchen ist es **0,500**, der Wert des Plans war ein Rundungsartefakt der Quadratur (Mittelpunktregel). Auf 0,45 ist die Beste Antwort 0,47, wie im Plan. Hedge mit fester Rate 1,0 landet nach 20.000 Runden nicht in 4 von 4, sondern in 3 von 4 Läufen bei 0,450 (auf drei Stellen) und im übrigen bei 0,500.
- **Q-Lernen weicht von der Vormessung ab** (andere Zufallsströme): im Zweitpreis 0,871 statt 0,909 (Plan), im Erstpreis 0,584 (Plan 0,577); die Richtung (Erstpreis über dem Gleichgewicht, Zweitpreis darunter) bleibt, die Zahlen nicht: Q-Lernen ist rasterempfindlich, Lernergebnisse stehen nur als Bandbreiten da.
- **Mischung: Hülle statt Wertgitter.** Das beste Verfahren der unregulären Mischung kommt aus der oberen konkaven Hülle der Erlöskurve (100.001 Stützstellen statt 400.001 der Vormessung); die Zahlen der Vormessung (9,00; 3 / 4 / 6 / 7 / 10 zusätzliche Bieter bei n = 1 / 2 / 3 / 5 / 10) bleiben, die Suite prüft die Stützstellenzahl mit 200.001.
- **Rechnung ohne scipy:** Gleichgewichtsgebote und Erlöse sind Trapezintegrale auf einem Gitter von 20.001 Punkten (alle Ansichten zusammen bei den Standardeinstellungen 0,33 s, bei 20 Bietern mit Exponentialverteilung 1,65 s; einmal gemessen, nicht als Test; die Vormessung brauchte für Erstpreis gegen Zweitpreis 171,6 s mit `quad` je Punkt), die Reserve eine Bisektion. `scipy` kommt nur als Orakel in den Tests vor.
- **Reiter umbenannt:** „Ein Bieter mehr gegen das beste Festangebot“ statt „Auktion gegen Verhandlung“. Bulow und Klemperer vergleichen mit dem optimalen Verfahren mit Selbstbindung (Auktion mit bestem Mindestgebot), nicht mit abwechselnden Angeboten wie im Verhandlungsstück; die Brücke dahin ist nur der Ultimatumcharakter des Festangebots (ein Bieter, ein Preis).
- **Neu aufgenommen, weil gemessen:** All-pay als dritte Regel (Gebot aus seiner Bedingung erster Ordnung, nicht aus der Erlösäquivalenz eingesetzt), die Voraussetzungen von Bulow–Klemperer (Regularität, Bieterwert ≥ Eigenwert), der Eigenwert des Verkäufers, die Negativkontrolle des Beste-Antwort-Prüfers.
- **Port und Name:** Port 9221 (die Vorschläge 9214 und 9219 des Plans sind belegt).

## Modell

Ein Slot, *n* Spediteure mit unabhängigen Werten *vᵢ* ~ *F* (Euro je Slot), ein Verkäufer mit Eigenwert *v₀* (was der unverkaufte Slot ihm bringt) und Mindestgebot *r*. Vier Verteilungen: Gleichverteilung auf 0 bis 100 Euro, Exponentialverteilung (Mittel 50 Euro, für die Gitterrechnung bei der Wahrscheinlichkeit 1 − 10⁻¹⁰ abgeschnitten), Beta(2, 2) auf 0 bis 100 Euro und eine **Mischung** (90 % gleichverteilt auf 0 bis 10 Euro, 10 % auf 90 bis 100 Euro), deren virtueller Wert *v* − (1 − *F*)/*f* nicht steigt (unregulär). **Zweitpreis:** ehrlich bieten ist dominant, Preis = max(*r*, zweithöchstes Gebot). **Erstpreis:** *b*(*v*) = *v* − ∫ᵣᵛ (*F*(*x*)/*F*(*v*))ⁿ⁻¹ d*x*. **All-pay** (ohne Mindestgebot): *b*(*v*) = ∫ *x* (*n* − 1) *F*(*x*)ⁿ⁻² *f*(*x*) d*x*. Der Zweitpreis-Erlös wird auf zwei Wegen gerechnet (A: virtuelle Werte, B: Ordnungsstatistik). Dazu Risikoscheu (Nutzen *x*^ρ, Gleichverteilung), das Mindestgebot als Myerson-Reserve φ(*r*\*) = *v₀*, das beste Verfahren im Quantilenraum (Hülle der Erlöskurve), asymmetrische Bieter (zwei Gleichverteilte mit verschiedener Obergrenze, geschlossenes Gleichgewicht) und lernende Bieter (Hedge, Exp3, Q-Lernen). Formeln, Lesehilfe und Symboltabelle stehen im 📐-Expander der App.

## Methodik

- **Kern ohne scipy:** Integrale per Trapezregel auf einem Gitter von 20.001 Punkten, Nullstellen per Bisektion, Hülle per monotoner Kette; die Einheiten (`second_price_outcome`, `spa_revenue_enum`, `fpa_bid_function` ...) sind kleine Funktionen ohne Zustand, geprüft an den Mini-Instanzen von Hand (`tests/conftest.py`, `tests/test_units.py`).
- **Orakel** (`tests/test_oracle_*.py`, anderer Rechenweg): Gleichgewichtsgebote gegen das scipy-Integral der Formel und gegen die Differentialgleichung *b*′ = (*n* − 1) *f*/*F* · (*v* − *b*); Zweitpreis-Erlös auf zwei Wegen gegen eine dritte Zerlegung mit scipy `quad`; Erstpreis und All-pay gegen den Zweitpreis; **Beste-Antwort-Gitter** (die Gegner bieten die Gleichgewichtsgebote, 6.001 Gebote werden geprüft) mit **Negativkontrolle** (falsche Gegnerstrategie liefert eine Lücke); Reserve gegen die Formel, die Bisektion und `minimize_scalar`; Wohlfahrtsverlust gegen `quad`; Risikoscheu gegen `minimize_scalar`; die Hülle gegen scipy `ConvexHull`, gegen den besten Festpreis (n = 1) und gegen einen Monte-Carlo-Lauf; die Asymmetrie gegen die Differentialgleichungen, `dblquad`, Beste-Antwort-Gitter beider Bieter, Simulation und die exakte Bruchrechnung des Myerson-Erlöses; das Klassenspiel des Lernens in Brüchen gegen `quad` und `dblquad`.
- **Messreihen** (`tools/make_tables.py`, `tools/make_lernen.py`): vorgerechnet in `data/tables.json` und `data/lernen.json`, mit Hash der Quellen; jeder Block hat einen eigenen SplitMix64-Strom aus seiner Bezeichnung; ± ist der Standardfehler über die Blöcke. Eine Einstellung der Seitenleiste rechnet live (Gitter, Hülle, kurze Simulationen), ein Lernlauf auf Knopfdruck.
- **Zufall:** `eza_rng.py` ist eine wörtliche Kopie von `ab-test-demo/abt_rng.py` (SplitMix64) mit Kopietest.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Erlöse in Euro je Slot; „Zelle“ der Simulation sind 4 Blöcke zu je 200.000 Auktionen (Asymmetrie: 400.000), Lernen 4 Läufe; ± ist der Standardfehler der Blöcke. Gitterwerte tragen den Fehler der Trapezregel (kleiner als 10⁻⁴ Euro in den gezeigten Stellen, geprüft gegen scipy).

| Nr. | Hypothese | Stand | Befund |
|---|---|---|---|
| H1 | Das Gleichgewichtsgebot im Erstpreis ist das Integral b(v) = v − ∫(F(x)/F(v))ⁿ⁻¹ dx, bei Gleichverteilung (n − 1)/n · v | **Ja** | Gebot beim Wert 50 Euro: 25,00 / 33,33 / 40,00 / 45,00 Euro bei n = 2 / 3 / 5 / 10; größte Beste-Antwort-Lücke 4,4e-07 Euro in 9 von 9 Zeilen (Erstpreis und All-pay, Werte 20 / 50 / 80 / 100 Euro); Negativkontrolle schlägt an (Lücken 0,52, 1,46, 2,68 Euro bei Wert 30, 60, 90); Differentialgleichung und scipy-Integral in der Suite |
| H2 | Erlösäquivalenz: Erstpreis = Zweitpreis = All-pay (gleich starke, risikoneutrale Bieter, unabhängige Werte, kein Mindestgebot) | **Ja**, mit diesen Bedingungen | Zweitpreis (zwei Wege), Erstpreis und All-pay gleich 100(n − 1)/(n + 1) in 9 von 9 Zeilen (n = 2 bis 10, größte Abweichung 3,7e-06 Euro): 33,33 Euro bei n = 2, 81,82 Euro bei n = 10; Simulation innerhalb von 5 Standardfehlern (plus 0,02 Euro) in 27 von 27 Zellen; Exponentiell, Beta(2, 2) und Mischung (n = 2 und 5): in 6 von 6 Zeilen gleich (größte Abweichung 2,2e-05 Euro) |
| H3 | Risikoscheu hebt den Erstpreis über den Zweitpreis, der Abstand schrumpft mit n | **Ja**, für Gleichverteilung | Verhältnis über 1 in 9 von 9 Zellen mit ρ < 1: 1,6000 bei n = 2 und ρ = 0,25 bis 1,0256 bei n = 10 und ρ = 0,75; größtes Verhältnis je n: 1,6000 / 1,1765 / 1,0811 bei n = 2 / 5 / 10; bei ρ = 1 gleich 1,0000 in 3 von 3 Zellen |
| H4 | Die Myerson-Reserve ist bei Gleichverteilung (100 + v₀)/2, unabhängig von n; der Gewinn des Mindestgebots schrumpft mit n | **Ja** | Formel, Bisektion von φ(r) = v₀ und Gittersuche stimmen in 15 von 15 Zellen überein (r\* = 50 / 60 / 70 Euro bei Eigenwert 0 / 20 / 40, n = 1 bis 10); Gewinn gegenüber r = v₀ bei Eigenwert 0: 25,0 / 6,25 / 0,78 / 0,011 % bei n = 2 / 3 / 5 / 10, bei Eigenwert 20 Euro 80,0 / 28,8 / 2,3 / 0,081 % bei n = 1 / 2 / 5 / 10; Wohlfahrtsverlust bei Eigenwert 0: 12,5 / 1,56 / 0,049 % des effizienten Werts bei n = 2 / 5 / 10; Erstpreis mit Mindestgebot gleich Zweitpreis mit Mindestgebot in 3 von 3 Zellen |
| H5 | Ein Bieter mehr schlägt das beste Verfahren mit n Bietern (Bulow–Klemperer) | **Ja**, nur mit den Voraussetzungen des Aufsatzes | regulär, Eigenwert 0: 30 von 30 Zeilen (Gleichverteilung, Exponentiell, Beta(2, 2); n = 1 bis 10), Verhältnis 1,3333 bei n = 1 bis 1,0184 bei n = 10 (Gleichverteilung); Exponentiell mit Eigenwert 20 Euro: 9 von 10, bei n = 1 31,23 gegen 32,33 Euro (Verhältnis 0,9661); Mischung (unregulär): 0 von 15 Zeilen, bei n = 1 4,53 gegen 9,00 Euro, zusätzliche Bieter 3 / 4 / 6 / 7 / 10 bei n = 1 / 2 / 3 / 5 / 10 |
| H6 | Asymmetrie: der Erstpreis ist ineffizient und bringt mehr als der Zweitpreis | **Ja**, für U[0, 100] gegen U[0, 100·w] | Erstpreis / Zweitpreis 1,0033 / 1,0218 / 1,1016 / 1,2976 bei w = 0,9 / 0,75 / 0,5 / 0,25 (gleich bei w = 1); falsche Zuteilung 2,37 % / 5,36 % / 8,33 % / 7,50 %, Wohlfahrtsverlust 0,10 % / 0,61 % / 1,89 % / 2,17 %; der Zweitpreis vergibt mit Geboten gleich den Werten nie falsch (per Konstruktion); Myerson-optimal 1,25- bis 2,48-fach der Zweitpreis-Erlös bei 12,5 % bis 19,4 % Wohlfahrtsverlust |
| H7 | Lernende Bieter (Hedge, Exp3, Q-Lernen) finden die Gleichgewichtsgebote | **Nein** (mit Standardrate; Hedge mit fester Rate 1,0 erreicht sie im Zweitpreis) | in 10 von 10 Zellen mit Standardrate weicht der letzte Gebotsfaktor ab (um mehr als 0,01 plus 3 Standardfehler); Erstpreis, zwei Bieter, Gleichgewicht 0,500: Hedge 0,456 ± 0,002, Exp3 0,422 ± 0,002, Q-Lernen 0,584 ± 0,005 |
| H8 | Die Erlösäquivalenz bleibt bei lernenden Bietern erhalten | **Nein** | je nach Regel liegt mal der Erst-, mal der Zweitpreis vorn (n = 2): Hedge Zweitpreis höher (0,3217 gegen 0,3026), Exp3 Erstpreis höher (0,2853 gegen 0,2750), Q-Lernen Erstpreis höher (0,3915 gegen 0,2789); n = 4: Hedge Zweitpreis höher (0,5824 gegen 0,5581), Exp3 gleich innerhalb der Streuung (0,5218 ± 0,0005 gegen 0,5216 ± 0,0007); Gleichgewichtserlös 0,3333 bei n = 2, 0,6000 bei n = 4 |
| H9 | Lernen führt zu Bid Shading auf Gleichgewichtshöhe | **Nein** | Hedge und Exp3: der letzte Gebotsfaktor liegt in 8 von 8 Zellen unter dem Gleichgewicht; Q-Lernen im Erstpreis darüber (0,584 ± 0,005), im Zweitpreis darunter (0,871 ± 0,009). Rasterfalle: auf dem Raster mit Schritt 0,05 sind 0,45 und 0,50 beide Beste Antwort auf sich selbst; Hedge mit fester Rate 1,0 endet nach 20.000 Runden in 3 von 4 Läufen bei 0,450 |
| H10 | Die Teilnahmeschwelle des zusätzlichen Bieters ist p\* = (n + 2)/2ⁿ⁺¹ (Gleichverteilung, Eigenwert 0) | **Ja**, bewiesen | Herleitung (siehe Korrekturen), in Brüchen für n = 1 bis 12 nachgerechnet; gemessene Schwelle 0,7500 / 0,5000 / 0,3125 / 0,1875 / 0,1094 / 0,0625 / 0,0352 / 0,0059 bei n = 1 / 2 / 3 / 4 / 5 / 6 / 7 / 10 |

**Erstpreis, Zweitpreis, All-pay bei Gleichverteilung** (Experiment E1; Gebot beim Wert 50 Euro im Erstpreis, Erlöse auf allen Wegen):

| n | Gebot bei Wert 50 | Formel 100(n−1)/(n+1) | Zweitpreis Weg A | Zweitpreis Weg B | Erstpreis | All-pay |
|---|---|---|---|---|---|---|
| 2 | 25,00 | 33,3333 | 33,3333 | 33,3333 | 33,3333 | 33,3333 |
| 3 | 33,33 | 50,0000 | 50,0000 | 50,0000 | 50,0000 | 50,0000 |
| 4 | 37,50 | 60,0000 | 60,0000 | 60,0000 | 60,0000 | 60,0000 |
| 5 | 40,00 | 66,6667 | 66,6667 | 66,6667 | 66,6667 | 66,6667 |
| 6 | 41,67 | 71,4286 | 71,4286 | 71,4286 | 71,4286 | 71,4286 |
| 7 | 42,86 | 75,0000 | 75,0000 | 75,0000 | 75,0000 | 75,0000 |
| 8 | 43,75 | 77,7778 | 77,7778 | 77,7778 | 77,7778 | 77,7778 |
| 9 | 44,44 | 80,0000 | 80,0000 | 80,0000 | 80,0000 | 80,0000 |
| 10 | 45,00 | 81,8182 | 81,8182 | 81,8182 | 81,8182 | 81,8182 |

Simulation und Beste-Antwort-Lücken (größter Gewinn, den ein Bieter mit Wert 20, 50, 80 oder 100 Euro durch Abweichen hätte):

| n | Sim. Zweitpreis | Sim. Erstpreis | Sim. All-pay | Lücke Erstpreis (Euro) | Lücke All-pay (Euro) |
|---|---|---|---|---|---|
| 2 | 33,31 ± 0,01 | 33,32 ± 0,01 | 33,34 ± 0,02 | 0,0e+00 | 7,1e-15 |
| 3 | 50,01 ± 0,03 | 50,00 ± 0,02 | 50,00 ± 0,03 | 1,7e-08 | 0,0e+00 |
| 4 | 59,98 ± 0,01 | 60,00 ± 0,01 | 59,98 ± 0,05 | 1,6e-08 | 0,0e+00 |
| 5 | 66,68 ± 0,02 | 66,66 ± 0,01 | 66,61 ± 0,07 | 5,6e-09 | 0,0e+00 |
| 6 | 71,40 ± 0,01 | 71,42 ± 0,01 | 71,37 ± 0,08 | 1,6e-09 | 0,0e+00 |
| 7 | 75,03 ± 0,02 | 75,00 ± 0,01 | 74,92 ± 0,04 | 3,1e-07 | 0,0e+00 |
| 8 | 77,80 ± 0,00 | 77,78 ± 0,01 | 77,71 ± 0,04 | 1,1e-08 | 0,0e+00 |
| 9 | 79,99 ± 0,01 | 80,00 ± 0,01 | 80,03 ± 0,04 | 4,4e-07 | 0,0e+00 |
| 10 | 81,82 ± 0,01 | 81,82 ± 0,01 | 81,80 ± 0,05 | 5,4e-09 | 0,0e+00 |

Andere Verteilungen (die Äquivalenz hängt nicht an der Gleichverteilung):

| Verteilung | n | Zweitpreis Weg A | Zweitpreis Weg B | Erstpreis | All-pay | Sim. Erstpreis | Lücke Erstpreis (Euro) |
|---|---|---|---|---|---|---|---|
| Exponentiell | 2 | 25,0000 | 25,0000 | 25,0000 | 25,0000 | 24,99 ± 0,02 | 2,9e-06 |
| Exponentiell | 5 | 64,1667 | 64,1667 | 64,1667 | 64,1667 | 64,15 ± 0,01 | 9,1e-07 |
| Beta(2, 2) | 2 | 37,1429 | 37,1429 | 37,1429 | 37,1429 | 37,14 ± 0,02 | 1,8e-08 |
| Beta(2, 2) | 5 | 62,1379 | 62,1379 | 62,1379 | 62,1379 | 62,16 ± 0,01 | 1,1e-07 |
| Mischung | 2 | 4,5333 | 4,5333 | 4,5333 | 4,5333 | 4,54 ± 0,00 | 3,0e-07 |
| Mischung | 5 | 14,1787 | 14,1787 | 14,1787 | 14,1787 | 14,18 ± 0,01 | 1,2e-05 |

Negativkontrolle des Prüfers (n = 2; die Gegner bieten fälschlich *w*²/100 statt *w*/2):

| Wert (Euro) | Lücke, Gegner bieten w²/100 | Lücke, Gegner bieten v/2 |
|---|---|---|
| 30 | 0,52 | 0,0e+00 |
| 60 | 1,46 | 0,0e+00 |
| 90 | 2,68 | 0,0e+00 |

**Risikoscheu** (Experiment E2; Nutzen *x*^ρ, Gleichverteilung):

| n | Exponent ρ | Gebotsfaktor | Erstpreis | Zweitpreis | Verhältnis | Sim. Erstpreis | Lücke (Euro) |
|---|---|---|---|---|---|---|---|
| 2 | 1,00 | 0,5000 | 33,33 | 33,33 | 1,0000 | 33,33 ± 0,02 | 0,0e+00 |
| 2 | 0,75 | 0,5714 | 38,10 | 33,33 | 1,1429 | 38,10 ± 0,02 | 0,0e+00 |
| 2 | 0,50 | 0,6667 | 44,44 | 33,33 | 1,3333 | 44,44 ± 0,01 | 0,0e+00 |
| 2 | 0,25 | 0,8000 | 53,33 | 33,33 | 1,6000 | 53,30 ± 0,01 | 0,0e+00 |
| 5 | 1,00 | 0,8000 | 66,67 | 66,67 | 1,0000 | 66,66 ± 0,01 | 0,0e+00 |
| 5 | 0,75 | 0,8421 | 70,18 | 66,67 | 1,0526 | 70,18 ± 0,01 | 0,0e+00 |
| 5 | 0,50 | 0,8889 | 74,07 | 66,67 | 1,1111 | 74,08 ± 0,01 | 0,0e+00 |
| 5 | 0,25 | 0,9412 | 78,43 | 66,67 | 1,1765 | 78,42 ± 0,01 | 0,0e+00 |
| 10 | 1,00 | 0,9000 | 81,82 | 81,82 | 1,0000 | 81,83 ± 0,01 | 0,0e+00 |
| 10 | 0,75 | 0,9231 | 83,92 | 81,82 | 1,0256 | 83,93 ± 0,01 | 0,0e+00 |
| 10 | 0,50 | 0,9474 | 86,12 | 81,82 | 1,0526 | 86,12 ± 0,02 | 0,0e+00 |
| 10 | 0,25 | 0,9730 | 88,45 | 81,82 | 1,0811 | 88,45 ± 0,01 | 0,0e+00 |

**Mindestgebot** (Experiment E3; Gleichverteilung, Nutzen in Euro je Slot; r\*: Formel (100 + v₀)/2, Bisektion von φ(r) = v₀, Gittersuche):

| Eigenwert | n | r* Formel | r* Bisektion | r* Gitter | Nutzen Weg A | Nutzen Weg B | Nutzen ohne Mindestgebot | Gewinn | Verkauf | Wohlfahrtsverlust |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 50,0 | 50,000 | 50,00 | 25,000 | 25,000 | 0,000 | – | 50,0 % | 25,000 % |
| 0 | 2 | 50,0 | 50,000 | 50,00 | 41,667 | 41,667 | 33,333 | 25,000 % | 75,0 % | 12,500 % |
| 0 | 3 | 50,0 | 50,000 | 50,00 | 53,125 | 53,125 | 50,000 | 6,250 % | 87,5 % | 6,250 % |
| 0 | 5 | 50,0 | 50,000 | 50,00 | 67,188 | 67,187 | 66,667 | 0,781 % | 96,9 % | 1,562 % |
| 0 | 10 | 50,0 | 50,000 | 50,00 | 81,827 | 81,827 | 81,818 | 0,011 % | 99,9 % | 0,049 % |
| 20 | 1 | 60,0 | 60,000 | 60,00 | 36,000 | 36,000 | 20,000 | 80,000 % | 40,0 % | 15,385 % |
| 20 | 2 | 60,0 | 60,000 | 60,00 | 47,733 | 47,733 | 37,067 | 28,777 % | 64,0 % | 11,155 % |
| 20 | 3 | 60,0 | 60,000 | 60,00 | 56,480 | 56,480 | 50,720 | 11,356 % | 78,4 % | 7,249 % |
| 20 | 5 | 60,0 | 60,000 | 60,00 | 68,222 | 68,222 | 66,694 | 2,290 % | 92,2 % | 2,801 % |
| 20 | 10 | 60,0 | 60,000 | 60,00 | 81,884 | 81,884 | 81,818 | 0,081 % | 99,4 % | 0,230 % |
| 40 | 1 | 70,0 | 70,000 | 70,00 | 49,000 | 49,000 | 40,000 | 22,500 % | 30,0 % | 7,759 % |
| 40 | 2 | 70,0 | 70,000 | 70,00 | 56,200 | 56,200 | 47,200 | 19,068 % | 51,0 % | 7,849 % |
| 40 | 3 | 70,0 | 70,000 | 70,00 | 62,005 | 62,005 | 55,120 | 12,491 % | 65,7 % | 6,514 % |
| 40 | 5 | 70,0 | 70,000 | 70,00 | 70,588 | 70,588 | 67,418 | 4,703 % | 83,2 % | 3,776 % |
| 40 | 10 | 70,0 | 70,000 | 70,00 | 82,178 | 82,178 | 81,825 | 0,431 % | 97,2 % | 0,735 % |

Andere Verteilungen:

| Verteilung | Eigenwert | n | r* Bisektion | r* Gitter | Nutzen Weg A | Nutzen Weg B |
|---|---|---|---|---|---|---|
| Exponentiell | 0 | 1 | 50,000 | 50,02 | 18,394 | 18,394 |
| Exponentiell | 0 | 2 | 50,000 | 50,02 | 33,405 | 33,405 |
| Exponentiell | 0 | 5 | 50,000 | 50,02 | 65,357 | 65,357 |
| Exponentiell | 20 | 1 | 70,000 | 70,00 | 32,330 | 32,330 |
| Exponentiell | 20 | 2 | 70,000 | 70,00 | 43,139 | 43,139 |
| Exponentiell | 20 | 5 | 70,000 | 70,00 | 68,724 | 68,724 |
| Beta(2, 2) | 0 | 1 | 42,154 | 42,16 | 25,997 | 25,997 |
| Beta(2, 2) | 0 | 2 | 42,154 | 42,16 | 41,365 | 41,365 |
| Beta(2, 2) | 0 | 5 | 42,154 | 42,16 | 62,232 | 62,232 |
| Beta(2, 2) | 20 | 1 | 51,685 | 51,69 | 35,042 | 35,042 |
| Beta(2, 2) | 20 | 2 | 51,685 | 51,69 | 45,489 | 45,489 |
| Beta(2, 2) | 20 | 5 | 51,685 | 51,69 | 62,642 | 62,642 |

Simulation gegen die beiden Rechenwege (Zweitpreis, Gleichverteilung):

| n | Eigenwert | Mindestgebot | Nutzen Weg B | Nutzen Weg A | Simulation |
|---|---|---|---|---|---|
| 2 | 20 | 50 | 46,667 | 46,667 | 46,672 ± 0,040 |
| 2 | 20 | 60 | 47,733 | 47,733 | 47,714 ± 0,028 |
| 2 | 20 | 70 | 46,400 | 46,400 | 46,399 ± 0,008 |
| 5 | 20 | 50 | 67,812 | 67,813 | 67,811 ± 0,009 |
| 5 | 20 | 60 | 68,222 | 68,222 | 68,184 ± 0,014 |
| 5 | 20 | 70 | 67,227 | 67,227 | 67,227 ± 0,011 |

Erstpreis mit Mindestgebot gleich Zweitpreis mit Mindestgebot:

| n | Mindestgebot | Erstpreis | Zweitpreis | Sim. Erstpreis | Gebot beim Wert r | Lücke (Euro) |
|---|---|---|---|---|---|---|
| 2 | 50 | 41,6667 | 41,6667 | 41,622 ± 0,023 | 50,000 | 0,0e+00 |
| 3 | 30 | 51,4850 | 51,4850 | 51,481 ± 0,028 | 30,000 | 2,1e-08 |
| 5 | 60 | 66,6667 | 66,6667 | 66,694 ± 0,011 | 60,000 | 6,2e-09 |

**Ein Bieter mehr gegen das beste Festangebot** (Experiment E4; bestes Verfahren mit n Bietern gegen Zweitpreis mit n + 1 Bietern, Mindestgebot gleich Eigenwert). Gleichverteilung, Eigenwert 0:

| n | Bestes Verfahren (n Bieter) | Einfach (n + 1 Bieter) | Verhältnis | Aussage gilt | Zusätzliche Bieter | Schwelle p* |
|---|---|---|---|---|---|---|
| 1 | 25,000 | 33,333 | 1,3333 | ja | 1 | 0,7500 |
| 2 | 41,667 | 50,000 | 1,2000 | ja | 1 | 0,5000 |
| 3 | 53,125 | 60,000 | 1,1294 | ja | 1 | 0,3125 |
| 4 | 61,250 | 66,667 | 1,0884 | ja | 1 | 0,1875 |
| 5 | 67,187 | 71,429 | 1,0631 | ja | 1 | 0,1094 |
| 6 | 71,652 | 75,000 | 1,0467 | ja | 1 | 0,0625 |
| 7 | 75,098 | 77,778 | 1,0357 | ja | 1 | 0,0352 |
| 8 | 77,821 | 80,000 | 1,0280 | ja | 1 | 0,0195 |
| 9 | 80,020 | 81,818 | 1,0225 | ja | 1 | 0,0107 |
| 10 | 81,827 | 83,333 | 1,0184 | ja | 1 | 0,0059 |

Exponentiell, Eigenwert 20 Euro (n = 1 verletzt die Aussage):

| n | Bestes Verfahren (n Bieter) | Einfach (n + 1 Bieter) | Verhältnis | Aussage gilt | Zusätzliche Bieter | Schwelle p* |
|---|---|---|---|---|---|---|
| 1 | 32,330 | 31,233 | 0,9661 | nein | 2 | 1,0976 |
| 2 | 43,139 | 43,660 | 1,0121 | ja | 1 | 0,9581 |
| 3 | 52,679 | 54,811 | 1,0405 | ja | 1 | 0,8088 |
| 4 | 61,151 | 64,376 | 1,0527 | ja | 1 | 0,6628 |
| 5 | 68,724 | 72,568 | 1,0559 | ja | 1 | 0,5307 |
| 6 | 75,533 | 79,665 | 1,0547 | ja | 1 | 0,4178 |
| 7 | 81,692 | 85,900 | 1,0515 | ja | 1 | 0,3251 |
| 8 | 87,293 | 91,451 | 1,0476 | ja | 1 | 0,2510 |
| 9 | 92,414 | 96,449 | 1,0437 | ja | 1 | 0,1928 |
| 10 | 97,120 | 100,994 | 1,0399 | ja | 1 | 0,1475 |

Mischung, Eigenwert 0 (die Aussage gilt in keiner Zeile):

| n | Bestes Verfahren (n Bieter) | Einfach (n + 1 Bieter) | Verhältnis | Aussage gilt | Zusätzliche Bieter | Schwelle p* |
|---|---|---|---|---|---|---|
| 1 | 8,999 | 4,533 | 0,5037 | nein | 3 | 1,9851 |
| 2 | 17,132 | 7,880 | 0,4600 | nein | 4 | 3,7645 |
| 3 | 24,483 | 11,011 | 0,4497 | nein | 6 | 5,3025 |
| 4 | 31,129 | 14,179 | 0,4555 | nein | 7 | 6,3515 |
| 5 | 37,140 | 17,441 | 0,4696 | nein | 7 | 7,0384 |
| 6 | 42,576 | 20,793 | 0,4884 | nein | 8 | 7,4983 |
| 7 | 47,495 | 24,209 | 0,5097 | nein | 9 | 7,8179 |
| 8 | 51,947 | 27,655 | 0,5324 | nein | 9 | 8,0478 |
| 9 | 55,977 | 31,101 | 0,5556 | nein | 9 | 8,2184 |
| 10 | 59,626 | 34,518 | 0,5789 | nein | 10 | 8,3479 |
| 15 | 73,342 | 50,448 | 0,6878 | nein | 11 | 8,6759 |
| 20 | 81,790 | 63,499 | 0,7764 | nein | 12 | 8,7646 |
| 30 | 90,402 | 80,715 | 0,8928 | nein | 13 | 8,6500 |
| 40 | 94,004 | 89,495 | 0,9520 | nein | 12 | 8,2549 |
| 60 | 96,583 | 95,786 | 0,9918 | nein | 9 | 6,4332 |

Die übrigen vorgerechneten Reihen (Exponentiell und Beta(2, 2) mit Eigenwert 0, Gleichverteilung und Beta mit Eigenwert 20 Euro) stehen in `data/tables.json` und in der App.

**Asymmetrie** (Experiment E5; zwei Bieter, Werte U[0, 100] und U[0, 100·w]):

| w | Zweitpreis | Erstpreis | Myerson | Erst / Zweit | Myerson / Zweit | falsche Zuteilung | Verlust Erstpreis | Verlust Myerson |
|---|---|---|---|---|---|---|---|---|
| 1,00 | 33,33 | 33,33 | 41,67 | 1,0000 | 1,250 | 0,00 % | 0,00 % | 12,5 % |
| 0,90 | 31,50 | 31,60 | 39,63 | 1,0033 | 1,258 | 2,37 % | 0,10 % | 12,6 % |
| 0,75 | 28,12 | 28,74 | 36,72 | 1,0218 | 1,306 | 5,36 % | 0,61 % | 13,2 % |
| 0,50 | 20,83 | 22,95 | 32,29 | 1,1016 | 1,550 | 8,33 % | 1,89 % | 15,4 % |
| 0,25 | 11,46 | 14,87 | 28,39 | 1,2976 | 2,477 | 7,50 % | 2,17 % | 19,4 % |

Kontrolle (Simulation, Beste-Antwort-Lücken beider Bieter, höchstes Gebot):

| w | Sim. Zweitpreis | Sim. Erstpreis | Sim. Myerson | Lücke stark (Euro) | Lücke schwach (Euro) | höchstes Gebot |
|---|---|---|---|---|---|---|
| 1,00 | 33,34 ± 0,02 | 33,33 ± 0,01 | 41,65 ± 0,01 | 0,0e+00 | 0,0e+00 | 50,00 |
| 0,90 | 31,49 ± 0,02 | 31,60 ± 0,01 | 39,64 ± 0,03 | 3,3e-14 | 0,0e+00 | 47,37 |
| 0,75 | 28,16 ± 0,02 | 28,74 ± 0,01 | 36,70 ± 0,02 | 0,0e+00 | 0,0e+00 | 42,86 |
| 0,50 | 20,83 ± 0,01 | 22,96 ± 0,00 | 32,31 ± 0,02 | 0,0e+00 | 0,0e+00 | 33,33 |
| 0,25 | 11,46 ± 0,00 | 14,87 ± 0,00 | 28,39 ± 0,03 | 0,0e+00 | 0,0e+00 | 20,00 |

**Lernende Bieter** (Experiment E6, gelb; Werte U[0, 1] = 100 Euro, Gebot = α · Wert, α auf dem Raster 0,05; Erlös in Einheiten von 100 Euro):

| Regel | n | Format | Runden | α Gleichgewicht | α letzter Iterierter | Erlös | Erlös Gleichgewicht |
|---|---|---|---|---|---|---|---|
| Hedge | 2 | Erstpreis | 20.000 | 0,500 | 0,456 ± 0,002 | 0,3026 ± 0,0011 | 0,3333 |
| Hedge | 2 | Zweitpreis | 20.000 | 1,000 | 0,961 ± 0,000 | 0,3217 ± 0,0019 | 0,3333 |
| Hedge | 4 | Erstpreis | 20.000 | 0,750 | 0,698 ± 0,002 | 0,5581 ± 0,0017 | 0,6000 |
| Hedge | 4 | Zweitpreis | 20.000 | 1,000 | 0,973 ± 0,000 | 0,5824 ± 0,0009 | 0,6000 |
| Exp3 | 2 | Erstpreis | 100.000 | 0,500 | 0,422 ± 0,002 | 0,2853 ± 0,0010 | 0,3333 |
| Exp3 | 2 | Zweitpreis | 100.000 | 1,000 | 0,858 ± 0,004 | 0,2750 ± 0,0011 | 0,3333 |
| Exp3 | 4 | Erstpreis | 100.000 | 0,750 | 0,635 ± 0,001 | 0,5218 ± 0,0005 | 0,6000 |
| Exp3 | 4 | Zweitpreis | 100.000 | 1,000 | 0,883 ± 0,001 | 0,5216 ± 0,0007 | 0,6000 |
| Q-Lernen | 2 | Erstpreis | 300.000 | 0,500 | 0,584 ± 0,005 | 0,3915 ± 0,0028 | 0,3333 |
| Q-Lernen | 2 | Zweitpreis | 300.000 | 1,000 | 0,871 ± 0,009 | 0,2789 ± 0,0051 | 0,3333 |

Horizont und Rate (Hedge, zwei Bieter; „Blöcke“ sind die letzten Gebotsfaktoren der 4 Läufe):

| Format | Runden | Rate | α Gleichgewicht | α letzter Iterierter | Blöcke |
|---|---|---|---|---|---|
| Erstpreis | 2.000 | Standardrate | 0,500 | 0,445 ± 0,007 | 0,458, 0,456, 0,438, 0,430 |
| Erstpreis | 20.000 | Standardrate | 0,500 | 0,456 ± 0,002 | 0,454, 0,454, 0,460, 0,454 |
| Erstpreis | 100.000 | Standardrate | 0,500 | 0,457 ± 0,001 | 0,458, 0,456, 0,458, 0,455 |
| Erstpreis | 2.000 | fest 1,0 | 0,500 | 0,454 ± 0,003 | 0,452, 0,459, 0,446, 0,462 |
| Erstpreis | 20.000 | fest 1,0 | 0,500 | 0,463 ± 0,012 | 0,500, 0,450, 0,450, 0,450 |
| Zweitpreis | 2.000 | Standardrate | 1,000 | 0,921 ± 0,001 | 0,922, 0,918, 0,922, 0,924 |
| Zweitpreis | 20.000 | Standardrate | 1,000 | 0,961 ± 0,000 | 0,962, 0,961, 0,961, 0,961 |
| Zweitpreis | 100.000 | Standardrate | 1,000 | 0,978 ± 0,000 | 0,978, 0,978, 0,978, 0,978 |
| Zweitpreis | 2.000 | fest 1,0 | 1,000 | 0,982 ± 0,000 | 0,982, 0,982, 0,982, 0,982 |
| Zweitpreis | 20.000 | fest 1,0 | 1,000 | 1,000 ± 0,000 | 1,000, 1,000, 1,000, 1,000 |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wo gemessen |
|---|---|---|
| **Risikoneutral** (Nutzen = Euro) | Bei Risikoscheu (Exponent 0,5, zwei Bieter) bringt der Erstpreis 44,44 statt 33,33 Euro im Zweitpreis, Verhältnis 1,3333; mit zehn Bietern nur noch 1,0526. | E2 |
| **Gleich starke Bieter** | Hat der zweite Bieter Werte nur bis 50 Euro (w = 0,5), bringt der Erstpreis 22,95 statt 20,83 Euro (Verhältnis 1,1016) und vergibt den Slot in 8,33 % der Auktionen nicht an den höheren Wert. | E5 |
| **Reguläre Verteilung** (der virtuelle Wert steigt) | Bei der Mischung gilt „ein Bieter mehr schlägt das beste Verfahren“ in 0 von 15 Zeilen; bei einem Bieter bringt das beste Verfahren 9,00 Euro, zwei Bieter ohne Mindestgebot 4,53 Euro. | E4 |
| **Bieterwert ≥ Eigenwert des Verkäufers** („serious bidder“) | Exponentiell verteilte Werte mit Eigenwert 20 Euro: bei einem Bieter 31,23 Euro (zwei Bieter) gegen 32,33 Euro (bestes Verfahren), die Aussage gilt in 9 von 10 Zeilen. | E4 |
| **Unabhängige Werte, ein Gut, keine Absprache, keine Teilnahmekosten** | Nicht gerechnet. Mehrere Plätze (Positionsauktion) rechnet diese Demo nicht: sie hat einen Slot. | – |
| **Der Verkäufer bindet sich** an Format und Mindestgebot | Nicht gerechnet: ein Verkäufer, der nach einem Fehlschlag nachverhandelt, ändert die Anreize der Bieter. | – |
| **Die Bieter spielen das Gleichgewicht** | Lernende Bieter landen mit Standardrate nicht dort: Hedge mit zwei Bietern im Erstpreis kommt nach 20.000 Runden auf den Gebotsfaktor 0,456 statt 0,5 (4 Läufe). Das ist ein Befund über die gewählten Lernregeln und das Raster, nicht über echte Spediteure. | E6 |
| **Literatur nur teilweise im Volltext gelesen** | Bulow und Klemperer (1996) wurde im Volltext gelesen (laut Einzelplan); Myerson (1981), Vickrey (1961), Riley und Samuelson (1981), Griesmer, Levitan und Shubik (1967) und Banchio und Skrzypacz (2022) nicht. Die Demo rechnet das beschriebene Modell nach und behauptet keine ihrer Aussagen. | – |

Weitere Grenzen: Die Bulow–Klemperer-Reihen rechnen das beste Verfahren als Hülle im Quantilenraum (bei der Mischung ein Verfahren mit Glättung der Erlöskurve); ob diese Rechnung der Konstruktion im Aufsatz entspricht, ist für die regulären Verteilungen gegen die Reserve aus φ(r\*) = v₀ geprüft, für die Mischung nur gegen den besten Festpreis bei einem Bieter, scipy `ConvexHull` und einen Monte-Carlo-Lauf. Das Myerson-optimale Verfahren der Asymmetrie gilt für diese Verteilungsfamilie. Die Rechnung mit Risikoscheu gilt für Gleichverteilung. Der Lernreiter prüft keine Gleichgewichtsgebote, er zeigt, was bestimmte Lernregeln mit bestimmten Raten und einem Raster tun.

## Literatur

Bulow und Klemperer, „Auctions versus Negotiations“, American Economic Review 86(1), 1996, S. 180–194: laut Einzelplan im Volltext gelesen (Theorem 1 und 2, Voraussetzungen unabhängige Signale, Risikoneutralität, „serious bidder“, Regularität). **Nicht im Volltext geprüft:** Myerson, „Optimal Auction Design“, Mathematics of Operations Research 6(1), 1981, S. 58–73 (nur Bibliographie und Sekundärauszug: Zweitpreis mit Mindestgebot bei gleichartigen Bietern); Vickrey 1961; Riley und Samuelson 1981; Erlösäquivalenz als Lehrbuchstoff; Griesmer, Levitan und Shubik 1967 und Maskin und Riley (asymmetrische Erstpreis-Gleichgewichte; die Formel hier ist selbst hergeleitet und gemessen); Banchio und Skrzypacz 2022 (arXiv 2202.05947, nur Abstract: Q-Lernen im Erstpreis ohne weitere Rückmeldung liefert stillschweigend abgestimmte, niedrige Gebote (Kollusion), im Zweitpreis nicht; hier **nicht reproduziert**, anderes Modell, und Q-Lernen liegt hier im Erstpreis über dem Gleichgewicht). Einordnung: der Stoff ist Lehrbuchstoff; neu ist die Zusammenstellung im Portfolio, nicht die Theorie.

## Tests

**339 Tests**, Laufzeit rund 7 Minuten seriell auf Windows mit Python 3.14 (den größten Teil brauchen die AppTest-Dateien, auf einem Rechner mit parallelen Läufen anderer Sitzungen; die Linux-CI mit Python 3.12 ist noch nicht gelaufen). Zusammensetzung:

- **Orakel (`tests/test_oracle_*.py`, vier Dateien):** Gebotsfunktionen gegen scipy-Integral und Differentialgleichung, Zweitpreis-Erlös auf drei Wegen, Erlösäquivalenz für alle vier Verteilungen, Beste-Antwort-Gitter mit Negativkontrolle, Reserve gegen Formel, Bisektion und `minimize_scalar`, Risikoscheu, Wohlfahrtsverlust, Simulation innerhalb von 5 Standardfehlern; die Hülle gegen `ConvexHull`, den besten Festpreis und Monte-Carlo; die Asymmetrie gegen Differentialgleichungen, `dblquad`, Beste-Antwort-Gitter, Simulation und exakte Brüche; das Klassenspiel des Lernens gegen `quad` und `dblquad`.
- **Einheiten an von Hand gerechneten Mini-Instanzen (`test_units.py`):** Zweitpreis-Ausgang, Aufzählung ({1, 2, 3}: 14/9 und 17/9), Verteilungen, Gebotsfunktionen, Erlöse, Reserve, Hülle, Asymmetrie, Lernschritte; Handwerte mit Rechnung in `tests/conftest.py`.
- **Eigenschafts- und Zweigtests (`test_properties.py`):** Umbenennen der Bieter, Skalieren, Monotonie in n, Gebot höchstens der Wert, Hülle über der Kurve, bestes Verfahren über jedem Mindestgebot; jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab (Erst gegen Zweit bei ρ < 1 und w < 1, gleich bei ρ = 1 und w = 1; Bulow–Klemperer gilt bei Gleichverteilung und versagt bei Mischung und Exponentiell mit Eigenwert; die Lernrate wirkt bei jeder Regel); gleicher Seed, gleiches Ergebnis, auch in einem zweiten Prozess mit anderem Hash-Seed.
- **Randfälle (`test_edge_cases.py`):** ein Bieter, genau zwei Bieter, Gleichstand, Mindestgebot am Rand des Trägers und in der Lücke der Mischung, Eigenwert über dem Mindestgebot, 20 Bieter für alle Verteilungen, kürzeste Lernläufe.
- **Verdikte und Presets (`test_verdicts.py`, `test_presets.py`):** Schwellen und jede Zahl im Text an handgesetzten Werten; je Preset die erwartete Verdikt-Zeile.
- **Messreihen (`test_measurements.py`):** Erzeuger-Hash, Zellen gegen den Code, einzelne Zellen neu gerechnet.
- **Claims (`test_claims.py`):** jede Zahl der README und der Preset-Hilfen; die README-Tabellen werden mit denselben Zeilenbauern der App erzeugt und im Test als Text in der README gesucht; Hypothesenzeilen H1 bis H10.
- **App (`test_app.py`, `test_layout.py`, `test_footer.py`, `test_runtime_imports.py`, `test_copies.py`):** jeder Knopf, jeder Regler an beiden Rändern und durch alle Auswahlwerte, alle Minima und alle Maxima zusammen, der Lernlauf am Maximum für jede Regel und jedes Format, Permalink mit unsinnigen Werten, Hervorhebung des aktiven Presets, Footer, Laufzeit-Importe, Kopietest des Zufallsgenerators.

Ein Mutationstest läuft für diese Demo nicht (Entscheidung des Betreibers, 2026-10-28).


## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-App: Hauptansicht, fünf Reiter, Grenzen, Formeln, PDF |
| `eza_distributions.py` | die vier Verteilungen (Dichte, Verteilungsfunktion, Quantil, virtueller Wert) |
| `eza_auction.py` | Kern: Zweitpreis-Ausgang, Gebotsfunktionen, Erlöse auf zwei Wegen, Reserve, Wohlfahrt, Beste-Antwort-Gitter, Simulation |
| `eza_hull.py` | Bulow–Klemperer: Erlöskurve, obere Hülle, bestes Verfahren, zusätzliche Bieter, Teilnahmeschwelle |
| `eza_asymmetry.py` | geschlossenes Gleichgewicht zweier ungleicher Bieter, Erlöse, falsche Zuteilung |
| `eza_learning.py` | Hedge, Exp3, Q-Lernen, das Klassenspiel in Brüchen |
| `eza_experiments.py` | Messreihen E1 bis E6, Erzeuger-Hash, Laden |
| `eza_evaluation.py` | Kennzahlen, Zeilenbauer der Tabellen, Verdikte als reine Funktionen |
| `eza_scenario.py`, `eza_constants.py`, `eza_presets.py`, `eza_layout.py` | Einstellungen, Grenzen und Presets, Permalink, fester Kopf |
| `eza_texts.py`, `eza_visualization.py`, `eza_pdf_export.py` | Texte (auch in dieser README), Abbildungen, PDF |
| `eza_rng.py` | SplitMix64, wörtliche Kopie von `ab-test-demo/abt_rng.py` |
| `data/tables.json`, `data/lernen.json` | vorgerechnete Reihen mit Erzeuger-Hash |
| `tools/make_tables.py`, `tools/make_lernen.py` | rechnen die Reihen neu |
| `tests/` | Einheiten, Orakel, Eigenschaften, Randfälle, Verdikte, Messreihen, Zahlen der README, App |

## Bewusst nicht umgesetzt

- **Mehrere Plätze oder Güter** (Positionsauktion, Bündel): nicht gerechnet, hier gibt es einen Slot.
- **Korrelierte Werte und gemeinsame Werte** (affiliated values, Fluch des Gewinners): die Bieter sind unabhängig.
- **Private Information auf beiden Seiten** (doppelte Auktion, Chatterjee–Samuelson): `myerson-satterthwaite-demo`.
- **Verhandlung mit Geheimnissen und Verzögerung**, **Teilnahmekosten**, **Absprache der Bieter**, **dynamische Auktionen** (aufsteigende Uhr, holländische Auktion): nicht gerechnet. Die englische Auktion kommt nur als Beschreibung des Zweitpreises vor.
- **Reproduktion von Banchio und Skrzypacz (2022)**: anderes Modell; der Befund dort ist nur als Abstract zitiert.
- **Echte Auktionsdaten**: keine.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Die Streamlit-Cloud-Einstellung der Python-Version steht beim Deploy in den App-Einstellungen; die CI nutzt Python 3.12. Die vorgerechneten Reihen liegen im Repo; nach einer Änderung am Rechencode `python tools/make_tables.py` und `python tools/make_lernen.py` (einige Minuten) ausführen.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
