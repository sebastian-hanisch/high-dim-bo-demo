"""PDF-Export des Live-Laufs (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF

import hdb_constants as C


def _num(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def build_pdf(settings, runs, optimum):
    """Einseitiger Bericht: Einstellungen und Regret je Verfahren nach dem Budget. Gibt die Bytes der PDF-Datei zurück."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Hohe Dimensionen: wann trägt Bayessche Optimierung noch?", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for k, v in (("Funktion", C.PROBLEM_SHORT["hartmann"]), ("Zahl der Eingaben", str(settings.dim)), ("Auswertungen", str(settings.n)), ("Zufalls-Seed", str(settings.seed))):
        pdf.cell(60, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Regret nach dem Budget (bester Wert minus Minimum)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for m, run in runs.items():
        pdf.cell(0, 6, f"{C.METHOD_LABELS[m]}: {_num(run.best_so_far[-1] - optimum, 3)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Ein einzelner Lauf zeigt nur, was bei diesem Seed geschah; die Studie der Demo mittelt über 10 feste Läufe je Verfahren und Dimension.")
    return bytes(pdf.output())
