# Hohe Dimensionen: wann trägt Bayessche Optimierung noch? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-high-dim-bo-demo.streamlit.app/)** -->

Fünftes Stück der **Bayessche-Optimierung-Linie** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [constrained-bo-demo](https://github.com/sebastian-hanisch/constrained-bo-demo) fort, wechselt aber das Beispiel: Statt des Terminals dienen zwei Testfunktionen mit
**wachsender Zahl der Eingaben**, weil sich die Dimension am Terminal nicht beliebig erhöhen lässt. Im Unterschied zu den Fall-Demos zeigt die Demo ein einziges Verfahren an einem
wachsenden Beispiel.

## Warum dieses Problem

Eine verbreitete Faustregel sagt: Bayessche Optimierung funktioniere bis etwa 10 bis 20 Eingaben, darüber nicht mehr. Die Demo prüft die Regel an zwei Funktionen, die das
Entscheidende trennen: bei **Hartmann6** wirken nur sechs Eingaben, alle weiteren sind ohne Wirkung; bei **Ackley** wirken **alle**. Die Kernfrage: **Findet das Modell, welche Eingaben
zählen, und was bleibt, wenn alle zählen?**

## Modell

- **Aufgabe:** Minimierung auf dem Einheitswürfel, rauschfrei, 100 Auswertungen. Hartmann6 (Minimum −3,32237) in 6, 12, 24 und 48 Eingaben (die ersten sechs wirken); Ackley (Minimum 0)
  in 2, 6, 12 und 24 Eingaben (alle wirken).
- **Ersatzmodell:** Gauß-Prozess mit Matérn-5/2-Kern, entweder mit **je einem Längenmaß je Eingabe** (ARD) oder mit **einem gemeinsamen**. Die Hyperparameter maximieren die
  Randwahrscheinlichkeit (L-BFGS-B, Längenmaße begrenzt auf 0,05 bis 10). Akquisition: erwartete Verbesserung, maximiert über 1 950 Kandidaten.
- **Verfahren:** Zufall, CMA-ES (Paket `cma`, nur für die Studie), BO mit ARD, BO mit gemeinsamem Längenmaß, BO mit ARD und größeren Start-Längenmaßen (0,25·√d, angelehnt an
  neuere Arbeiten), BO mit Vertrauensbereich (TuRBO-artig). Startdesign min(d + 2, 30) Läufe.
- **Live-Teil:** Hartmann6 mit 6, 8, 10 oder 12 Eingaben, bis 40 Auswertungen; die gelernten Längenmaße je Eingabe werden gezeigt. Die Studie ist vorgerechnet
  (`precomputed_study.json`, 480 Läufe, `generate_study.py`).
- Mathematik im 📐-Expander der App.

## Methodik

Je Zelle (Funktion, Zahl der Eingaben, Verfahren) **10 feste Läufe** (Seeds 700000 bis 700009), gleiche Seeds für alle Verfahren (gepaarter Vergleich). Verglichen wird der Regret
(bester Wert minus Minimum) nach 25, 50 und 100 Auswertungen. Unterschiede werden mit dem Vorzeichentest geprüft (`scipy.stats.binomtest`); bei zehn Läufen trennt er nur klare
Unterschiede (neun von zehn oder mehr). „Kein Unterschied nachweisbar“ heißt nicht „kein Unterschied“.

## Befunde (gemessen, keine Behauptungen)

Median des Regrets nach 100 Auswertungen:

| Eingaben | Zufall | CMA-ES | BO, ARD | BO, gemeinsames Längenmaß | BO, größere Start-Längenmaße | BO, Vertrauensbereich |
|---|---|---|---|---|---|---|
| Hartmann6, 6 | 1,62 | 0,59 | 0,007 | 0,007 | 0,065 | 0,000 |
| Hartmann6, 12 | 1,39 | 0,92 | 0,121 | 0,016 | 0,006 | 0,002 |
| Hartmann6, 24 | 1,39 | 0,82 | 0,123 | 0,201 | 0,010 | 0,121 |
| Hartmann6, 48 | 1,27 | 0,56 | 0,129 | 0,202 | 0,067 | 0,192 |
| Ackley, 2 | 2,72 | 0,18 | 0,012 | 0,022 | 0,017 | 0,002 |
| Ackley, 6 | 4,93 | 3,74 | 1,87 | 0,61 | 0,97 | 3,09 |
| Ackley, 12 | 6,49 | 6,26 | 3,98 | 2,70 | 4,36 | 3,84 |
| Ackley, 24 | 7,52 | 7,36 | 5,63 | 4,84 | 5,12 | 6,02 |

| Frage | Befund |
|---|---|
| Hört BO bei 10 bis 20 Eingaben auf zu wirken? | **Nicht an diesen Funktionen.** Hartmann6 mit 48 Eingaben: BO mit ARD 0,129, Zufall 1,27, CMA-ES 0,56. BO ist in 10 von 10 Läufen besser als Zufall (p = 0,002) und besser als CMA-ES. |
| Blendet das Modell unwichtige Eingaben aus? | **Ja.** Mittleres gelerntes Längenmaß der unwirksamen Eingaben liegt bei 12, 24 und 48 Eingaben bei 10,0 (Obergrenze), das der wirksamen bei 2,0 bis 2,3. |
| Was bleibt, wenn alle Eingaben wirken? | Der Vorsprung gegenüber dem Zufall **schrumpft** mit der Dimension (Verhältnis des Regrets BO zu Zufall bei Ackley 2, 6, 12, 24 Eingaben steigt), bleibt aber bei 24 Eingaben sichtbar (ARD 9 von 10 Läufen besser als Zufall, p = 0,02). Absolut bleibt der Regret hoch (4,8 bis 5,6 von Zufall 7,5). |
| Hilft ein gemeinsames Längenmaß, größere Start-Längenmaße oder ein Vertrauensbereich? | **Ab 6 Eingaben bei zehn Läufen kein gesicherter Unterschied zu ARD auf Ackley** (bei 2 Eingaben ist der Vertrauensbereich besser, 9 von 10). Das gemeinsame Längenmaß liegt dort in den meisten Zellen vorn (6 Eingaben 0,61 gegen 1,87), p ≥ 0,1. Der Vertrauensbereich ist bei Hartmann6 mit 6 Eingaben besser (9 von 10, p = 0,02), bei 48 Eingaben schlechter (1 von 10). |
| Was kostet das Modell? | ARD braucht bei 48 Eingaben im Median 144 s je Lauf, das gemeinsame Längenmaß 11 s (ein Rechner, ein Kern; nur die Größenordnung zählt). |

Jede Zahl ist in `tests/test_claims.py` nachgerechnet.

## Ehrliche Grenzen

- **Zwei glatte Testfunktionen**, rauschfrei, auf dem Einheitswürfel. Hartmann6 hat eine einfache Struktur (sechs wirkende Eingaben); auf einer Aufgabe, bei der viele Eingaben
  schwach wirken, kann das Bild anders sein. Die Demo zeigt, dass die Faustregel an diesen Funktionen nicht gilt, nicht dass sie nie gilt.
- **Zehn Läufe je Zelle:** die Zellen trennen nur klare Unterschiede. Die Reihenfolge der drei BO-Varianten auf Ackley ist **nicht gesichert**.
- **Eigene, einfache Implementierung** (Kandidatenmenge, begrenzte Hyperparameter-Suche, höchstens 60 Schritte der Anpassung). Zahlen anderer Bibliotheken können abweichen.
  Die Literatur (Hvarfner, Hellsten, Nardi 2024; Xu u. a. 2025) kommt zu einem ähnlichen Ergebnis wie die Demo, mit anderen Implementierungen; sie ist hier **Gegenposition zur
  Faustregel**, kein Beleg für die eigenen Zahlen.
- **Das Startdesign** (d + 2 Läufe, höchstens 30) verbraucht bei 48 Eingaben 30 der 100 Auswertungen; bei 25 Auswertungen ist bei 24 und 48 Eingaben noch kaum etwas gelernt.
- **Keine ganzzahligen, kategorialen oder bedingten Eingaben** und kein Rauschen; beides kommt in der Praxis meist hinzu.
- **Rechenzeit** ist rechnerabhängig; die Zielfunktion kostet hier Mikrosekunden. In der Praxis ist die teure Auswertung die Regel, dann zählt das Modell weniger.

## Verifikation

- **Kern:** Kern, Erwartete Verbesserung und Vertrauensbereich gegen Handrechnung auf Mini-Instanzen; Interpolation rauschfreier Daten; ARD erkennt die unwirksame Eingabe.
- **Orakel:** Gauß-Prozess-Vorhersage gegen scikit-learn bei festen Hyperparametern (Mittelwert und Standardabweichung, 8 Konfigurationen); erwartete Verbesserung gegen numerische
  Integration; Vorzeichentest gegen die Binomialformel von Hand; Studie auf Vollständigkeit und Monotonie des besten Werts geprüft.
- **Aussagen:** `tests/test_claims.py` rechnet jede README-Zahl über dieselben Auswertungsfunktionen wie die App nach.
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `hdb_algorithm.py` | Kern: Kern, Gauß-Prozess, erwartete Verbesserung, Vertrauensbereich, Suchverfahren |
| `hdb_scenario.py` | Hartmann6 (eingebettet) und Ackley |
| `hdb_evaluation.py` | Läufe, vorgerechnete Studie, gepaarte Statistik, Längenmaße |
| `hdb_constants.py` | Regler-Grenzen, Verfahren, Farben, Presets |
| `hdb_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `hdb_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `hdb_pdf_export.py` | PDF-Auswertung des Live-Laufs |
| `generate_study.py` | erzeugt `precomputed_study.json` (CMA-ES braucht `requirements-dev.txt`) |
| `precomputed_study.json` | 480 Läufe der Studie |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scikit-learn, cma |
| `tests/` | Handrechnung, Orakel, Aussagen, Presets, AppTest-Smoke-Test, Footer |

## Bewusst nicht umgesetzt

- Zufällige Einbettungen (REMBO u. Ä.) und additive Modelle: andere Struktur als der Vergleich hier.
- Rauschen, gemischte Eingaben und Bibliotheksvergleich (BoTorch): eigene Stücke bzw. eigene Messung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
