"""Hohe Dimensionen: wann trägt Bayessche Optimierung noch? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Linie Bayessche Optimierung. Eine einzelne Frage: Wie wächst die Schwierigkeit mit der Zahl der Eingaben, und was hängt davon ab, wie viele davon wirken?
Live: Hartmann6 mit unwirksamen Zusatzeingaben; Studie: Hartmann6 bis 48 Eingaben, Ackley (alle wirksam) bis 24.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import hdb_constants as C
import hdb_evaluation as ev
import hdb_scenario as S
from hdb_evaluation import Settings
from hdb_pdf_export import build_pdf
from hdb_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from hdb_visualization import build_dimension_chart, build_lengthscale_chart, build_live_chart, build_study_chart

st.set_page_config(page_title="Hohe Dimensionen – Sebastian Hanisch", layout="wide")


def de(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


@st.cache_data(show_spinner=False)
def _live(settings, methods):
    return {m: ev.run_one("hartmann", settings.dim, m, settings.seed, settings.n) for m in methods}


st.title("📈 Hohe Dimensionen: wann trägt Bayessche Optimierung noch?")
st.markdown(
    """
Eine verbreitete Faustregel sagt: Bayessche Optimierung funktioniere bis etwa 10 bis 20 Eingaben, darüber nicht mehr. Die Demo prüft die Regel an zwei Testfunktionen. **Hartmann6** hat
sechs wirkende Eingaben, alle weiteren sind **ohne Wirkung**: kommt das Modell damit klar, bleibt die Aufgabe im Kern sechsdimensional. **Ackley** wirkt in **allen** Eingaben: dort hilft
nichts beim Ausblenden. Verglichen werden Zufall, **CMA-ES**, BO mit je einem Längenmaß je Eingabe (**ARD**), BO mit **einem gemeinsamen** Längenmaß, BO mit größeren Start-Längenmaßen
(angelehnt an neuere Arbeiten) und BO mit **Vertrauensbereich**. Die Demo ist das fünfte Stück der Linie **Bayessche Optimierung**; im Unterschied zu den Fall-Demos zeigt sie ein einziges
Verfahren an einem wachsenden Beispiel.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was sich mit der Dimension ändert", expanded=True):
    st.markdown(
        """
1. **Der Raum wächst:** Zufälliges Absuchen trifft die Mulde in 6 Eingaben mit einer bestimmten Wahrscheinlichkeit; in 48 Eingaben sind die Eingaben, die zählen, aber immer noch nur 6:
   die **wirksame Dimension** entscheidet, nicht die Zahl der Eingaben.
2. **Das Modell muss lernen, welche Eingaben zählen:** je Eingabe ein Längenmaß (ARD) heißt je Eingabe ein Parameter aus wenigen Beobachtungen. Ein großes Längenmaß bedeutet „diese Eingabe ist
   unwichtig“.
3. **Das Startdesign wächst mit:** Hier d + 2 Läufe (höchstens 30). Bei Budgets unter dieser Zahl lernt noch nichts.
4. **Der Rechenaufwand wächst:** Die Anpassung von d + 2 Hyperparametern auf n Beobachtungen kostet; die Studie zeigt Sekunden je Lauf.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Live-Lauf (Hartmann6)**")
    dim = st.select_slider("Zahl der Eingaben", C.LIVE_DIMENSIONS["hartmann"], key="dim_select",
                           help="Hartmann6 hat sechs wirkende Eingaben; alle weiteren sind ohne Wirkung und nur dazugefügt.")
    n = st.slider("Auswertungen", *bounds("n_slider"), key="n_slider", step=1, help=f"Budget des Live-Laufs (höchstens {C.LIVE_MAX_N}); das Startdesign nutzt min(d + 2, 30) davon.")
    method = st.radio("Verfahren", C.LIVE_METHODS, format_func=lambda m: C.METHOD_LABELS[m], key="method_radio",
                      help="Der Live-Lauf rechnet das gewählte Verfahren und den Zufall mit demselben Seed.")
    seed = st.number_input("Zufalls-Seed des Laufs", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"dim_select": int(dim), "n_slider": int(n), "method_radio": method, "seed_input": int(seed)})

settings = Settings(problem="hartmann", dim=int(dim), n=int(n), seed=int(seed))
methods = ("random",) if method == "random" else ("random", method)
runs = _live(settings, methods)
opt = C.GLOBAL_MIN["hartmann"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Hartmann6 in **{settings.dim} Eingaben** ({settings.dim - 6} ohne Wirkung), {settings.n} Auswertungen. Regret = bester gefundener Wert minus das Minimum ({de(opt, 5)}); 0 wäre das Optimum.")
cols = st.columns(len(runs))
for col, (m, r) in zip(cols, runs.items()):
    col.metric(C.METHOD_LABELS[m], de(r.best_so_far[-1] - opt, 3), help="Regret nach dem Budget.")
st.plotly_chart(build_live_chart(runs, opt), width="stretch", key="live_chart")

bo_run = runs.get(method)
if bo_run is not None and bo_run.lengthscales:
    st.markdown("**Was das Modell über die Eingaben gelernt hat** – Längenmaß je Eingabe am Ende des Laufs")
    if method == "bo_iso":
        st.info(f"Ein gemeinsames Längenmaß für alle Eingaben: **{de(bo_run.lengthscales[0], 2)}**. Das Modell kann keine Eingabe ausblenden.")
    else:
        st.plotly_chart(build_lengthscale_chart(bo_run.lengthscales, 6, settings.dim), width="stretch", key="ls_chart")
        act = sum(bo_run.lengthscales[:6]) / 6
        inact = sum(bo_run.lengthscales[6:]) / max(settings.dim - 6, 1) if settings.dim > 6 else None
        if inact is not None:
            st.caption(f"Mittel der gelernten Längenmaße: wirksame Eingaben (grün) **{de(act)}**, unwirksame (rot) **{de(inact)}**. Die Obergrenze des Modells ist {C.LS_BOUNDS[1]:g}: "
                       "dort „weiß“ das Modell nur noch nichts von der Eingabe.")
st.download_button("📄 Auswertung als PDF", data=build_pdf(settings, runs, opt), file_name="hohe_dimensionen_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Findet das Modell, welche Eingaben zählen?")
sumH = {d: ev.length_scales_summary("hartmann", d) for d in C.DIMENSIONS["hartmann"][1:]}
st.info(
    "In der Studie (10 feste Läufe, BO mit je einem Längenmaß je Eingabe, Hartmann6 mit 12, 24 und 48 Eingaben) liegen die gelernten Längenmaße der **unwirksamen** Eingaben im Mittel bei "
    + ", ".join(f"{de(sumH[d]['inactive'], 1)}" for d in sumH) + f", die der **wirksamen** bei " + ", ".join(f"{de(sumH[d]['active'], 1)}" for d in sumH)
    + ". **Das Modell blendet die unwichtigen Eingaben aus**, auch bei 48 Eingaben und 100 Auswertungen."
)
st.caption("Ein einzelner Lauf zeigt nur, was bei diesem Seed geschah; die Experimente mitteln über 10 feste Läufe je Verfahren und Dimension.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

cp = st.select_slider("Prüfpunkt der Studie (Auswertungen)", C.CHECKPOINTS, value=C.CHECKPOINTS[-1], key="cp_select", help="An welcher Zahl von Auswertungen die Studie verglichen wird.")

st.subheader("🔬 Experiment 1: Hartmann6 mit wachsend vielen Eingaben ohne Wirkung")
st.caption(f"Eingaben 6, 12, 24, 48 (wirksam sind immer die ersten sechs); Median des Regrets über 10 feste Läufe nach {cp} Auswertungen. Der Zufall ist von der Zahl der Eingaben praktisch unabhängig: "
           "Die unwirksamen Eingaben ändern den Funktionswert nicht.")
summ_h = {d: ev.table("hartmann", d) for d in C.DIMENSIONS["hartmann"]}
st.plotly_chart(build_dimension_chart(summ_h, cp, ("random", "cma", "bo_ard", "bo_iso", "bo_scaled", "bo_tr")), width="stretch", key=f"dim_hartmann_{cp}")
d48 = summ_h[48]
st.info(
    f"Bei 48 Eingaben und {cp} Auswertungen: Zufall {de(d48['random'][cp]['median'])}, CMA-ES {de(d48['cma'][cp]['median'])}, BO mit ARD {de(d48['bo_ard'][cp]['median'], 3)}, BO mit gemeinsamem "
    f"Längenmaß {de(d48['bo_iso'][cp]['median'], 3)}, mit größeren Start-Längenmaßen {de(d48['bo_scaled'][cp]['median'], 3)}, mit Vertrauensbereich {de(d48['bo_tr'][cp]['median'], 3)}. "
    + ("**Bei 25 Auswertungen ist bei 24 und 48 Eingaben noch kaum etwas gelernt**: das Startdesign (d + 2, höchstens 30 Läufe) füllt das ganze Budget. " if cp == 25 else "")
    + "Die Faustregel „ab 10 bis 20 Eingaben versagt es“ ist an dieser Funktion **nicht** bestätigt: es kommt auf die **wirksame** Dimension an."
)

st.subheader("🔬 Experiment 2: Ackley, alle Eingaben wirksam")
st.caption(f"Eingaben 2, 6, 12, 24, alle wirksam; Median des Regrets über 10 feste Läufe nach {cp} Auswertungen. Hier gibt es nichts auszublenden.")
summ_a = {d: ev.table("ackley", d) for d in C.DIMENSIONS["ackley"]}
st.plotly_chart(build_dimension_chart(summ_a, cp, ("random", "cma", "bo_ard", "bo_iso", "bo_scaled", "bo_tr")), width="stretch", key=f"dim_ackley_{cp}")
a24 = summ_a[24]
stat24 = ev.paired_stats("ackley", 24, "bo_ard", "random", cp)
st.info(
    f"Bei 24 Eingaben und {cp} Auswertungen: Zufall {de(a24['random'][cp]['median'])}, BO mit ARD {de(a24['bo_ard'][cp]['median'])}, gemeinsames Längenmaß {de(a24['bo_iso'][cp]['median'])}. "
    f"BO mit ARD ist in {stat24['better']} von {stat24['n']} Läufen besser als der Zufall. **Der Vorsprung schrumpft mit der Dimension**, bleibt bei 100 Auswertungen aber sichtbar; "
    "absolut liegt der Regret weit vom Optimum."
)

st.subheader("🔬 Experiment 3: Welche Stellschraube hilft?")
sp = st.radio("Funktion", C.PROBLEMS, format_func=lambda p: C.PROBLEM_SHORT[p], key="study_problem", horizontal=True)
sd = st.select_slider("Zahl der Eingaben", C.DIMENSIONS[sp], key=f"study_dim_{sp}")
rows = []
for key in ("bo_iso", "bo_scaled", "bo_tr", "cma", "random"):
    s = ev.paired_stats(sp, sd, key, "bo_ard", cp)
    rows.append(f"| {C.METHOD_LABELS[key]} | {s['better']} | {s['worse']} | {s['equal']} | {de(s['median_diff'], 3)} | {de(s['p'], 3)} | {ev.verdict_of(s)} |")
st.markdown(
    f"**{C.PROBLEM_SHORT[sp]}, {sd} Eingaben, {cp} Auswertungen:** gepaarter Vergleich gegen BO mit ARD (10 Läufe; Differenz = Variante minus ARD, negativ ist besser):\n\n"
    "| Variante | besser | schlechter | gleich | Median der Differenz | p | Urteil |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows)
)
st.plotly_chart(build_study_chart(ev.curves(sp, sd)), width="stretch", key=f"study_curves_{sp}_{sd}")
st.info("Der Vorzeichentest über **zehn** Läufe trennt nur klare Unterschiede (zehn von zehn oder neun von zehn). Auf der symmetrischen Ackley-Funktion liegt das gemeinsame Längenmaß in den "
        "meisten Läufen vor ARD (weniger Parameter, die Funktion ist in allen Eingaben gleich), gesichert ist das bei zehn Läufen nicht (ab 6 Eingaben). Auf Hartmann6 mit unwirksamen Eingaben ist "
        "zwischen ARD und gemeinsamem Längenmaß nichts nachweisbar. Der Vertrauensbereich bringt keinen durchgehenden Vorteil: bei 6 Eingaben besser, bei 48 schlechter.")

st.subheader("🔬 Experiment 4: Was kostet es an Rechenzeit?")
st.caption("Median der Rechenzeit je Lauf mit 100 Auswertungen in Sekunden (ein Rechner, ein Kern; nur die Größenordnung trägt, die Zeit hängt von Rechner und Bibliotheken ab).")
rows_t = []
for p in C.PROBLEMS:
    for d in C.DIMENSIONS[p]:
        t = ev.table(p, d)
        rows_t.append(f"| {C.PROBLEM_SHORT[p]} | {d} | " + " | ".join(f"{de(t[m]['seconds'], 1)}" for m in ("bo_ard", "bo_iso", "bo_scaled", "bo_tr", "cma")) + " |")
st.markdown("| Funktion | Eingaben | BO mit ARD | gemeinsames Längenmaß | größere Start-Längenmaße | Vertrauensbereich | CMA-ES |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows_t))
st.info("Je Eingabe ein Längenmaß verteuert die Anpassung deutlich (bei 48 Eingaben über zwei Minuten je Lauf); ein gemeinsames Längenmaß bleibt im Sekundenbereich. Eine Auswertung der "
        "Zielfunktion kostet hier Mikrosekunden: **in der Praxis ist die Simulation teuer, nicht das Modell**; das Modell zählt erst, wenn die Auswertung billig oder die Dimension sehr hoch ist.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Wenige Eingaben wirken** | Wirken viele Eingaben gleichzeitig (Ackley), schrumpft der Vorsprung mit der Dimension; hundert Auswertungen reichen dann nicht für ein gutes Ergebnis. | Strukturannahmen: additive Modelle, Einbettungen in kleine Räume |
| **Die Funktion ist rauschfrei** | Die Studie ist rauschfrei; mit Rauschen kommt eine weitere Fehlerquelle der Hyperparameter hinzu (siehe Verrauschte Auswertungen). | Rauschmodelle |
| **Die Eingaben sind stetig und beschränkt** | Ganzzahlige, kategoriale und bedingte Eingaben, wie sie in der Praxis häufig sind, brauchen andere Kerne. | Gemischte Räume |
| **Zehn Läufe je Zelle** | Die Zellen trennen nur klare Unterschiede (neun von zehn Läufe); „kein Unterschied nachweisbar“ heißt nicht „kein Unterschied“. | Mehr Läufe |
| **Eine Funktionsfamilie** | Hartmann6 und Ackley sind glatte Testfunktionen mit einer wohldefinierten Struktur. Auf Ihrer Aufgabe kann das Bild anders sein. | Eigene Messung |
| **Eine Implementierung** | Eigene, einfache Fassung (Kandidatenmenge, begrenzte Hyperparameter-Suche); Zahlen anderer Bibliotheken können abweichen. | Vergleich mit BoTorch o. Ä. |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Ersatzmodell.** Gauß-Prozess mit Matérn-5/2-Kern; mit Längenmaßen $\ell_1, \dots, \ell_d$ (ARD) oder einem gemeinsamen $\ell$:

$$k(x, x') = \sigma_f^2\Bigl(1 + \sqrt5\, r + \tfrac53 r^2\Bigr)e^{-\sqrt5\, r}, \qquad r = \sqrt{\textstyle\sum_{i=1}^{d} (x_i - x'_i)^2/\ell_i^2}$$

**Ausblenden.** Für $\ell_i \to \infty$ geht der Beitrag der Eingabe $i$ zu $r$ gegen null: die Funktion wird in dieser Eingabe konstant. Das Modell lernt $\ell_i$ aus der Randwahrscheinlichkeit

$$\log p(y \mid X) = -\tfrac12 y^\top K^{-1} y - \tfrac12 \log|K| - \tfrac n2 \log 2\pi$$

**Erwartete Verbesserung** (Minimierung, $z = (f^\* - \mu)/s$): $\mathrm{EI}(x) = (f^\* - \mu)\,\Phi(z) + s\,\varphi(z)$.

**Vertrauensbereich.** Kandidaten nur im Kasten $\prod_i [x^\*_i - h,\, x^\*_i + h]$ um die beste Beobachtung $x^\*$; $h$ startet bei 0,4, wird nach 3 Erfolgen in Folge mit 1,6
multipliziert (höchstens 0,8) und nach $\max(4, d/2)$ Misserfolgen halbiert (mindestens 0,02).

**Größere Start-Längenmaße.** Der Optimierer der Hyperparameter startet bei $\ell_i = 0{,}25\sqrt d$ statt bei $0{,}4$; die Begrenzung $[0{,}05, 10]$ bleibt.

**Startdesign.** $\min(d + 2, 30)$ gleichverteilte Läufe.

**Literatur.** C. Hvarfner, E. O. Hellsten, L. Nardi (2024), *Vanilla Bayesian Optimization Performs Great in High Dimensions*, ICML (Längenmaß-Prior wächst mit √d; Gegenposition zur
Faustregel); D. Eriksson, M. Pearce, J. Gardner, R. D. Turner, M. Poloczek (2019), *Scalable Global Optimization via Local Bayesian Optimization*, NeurIPS (Vertrauensbereiche);
Z. Wang, F. Hutter, M. Zoghi, D. Matheson, N. de Freitas (2016), *Bayesian Optimization in a Billion Dimensions via Random Embeddings*, JAIR 55 (Einbettungen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
