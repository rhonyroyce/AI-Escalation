"""
Pulse Dashboard - PDF Executive One-Pager
===========================================
Generates a dark-themed PDF executive summary using fpdf2.
Used by: 1_Executive_Summary.py.
"""
import io
from typing import Optional

from fpdf import FPDF


class PulsePDF(FPDF):
    """Custom FPDF subclass with dark-theme header/footer."""

    def __init__(self, title: str = "Pulse Executive Summary"):
        super().__init__()
        self._title = title

    def header(self):
        self.set_fill_color(10, 25, 41)  # #0a1929
        self.rect(0, 0, 210, 20, 'F')
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(224, 224, 224)
        self.set_y(5)
        self.cell(0, 10, self._title, align='C')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'CSE Intelligence Platform | Page {self.page_no()}', align='C')


def generate_executive_pdf(
    avg_pulse: float,
    target: float,
    variance: float,
    green_pct: float,
    red_count: int,
    total_entries: int,
    week: int,
    year: int,
    worst_dim: str,
    worst_dim_score: float,
    recommendations: list[dict],
    scr: Optional[dict] = None,
    chart_image_bytes: Optional[bytes] = None,
) -> bytes:
    """Generate a one-page executive PDF and return as bytes."""
    pdf = PulsePDF(f"Pulse Executive Summary - W{week}, {year}")
    pdf.add_page()

    # ── KPI row ──
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(224, 224, 224)
    pdf.set_fill_color(0, 30, 60)  # #001e3c

    kpis = [
        (f"Pulse: {avg_pulse:.1f}", f"vs {target:.0f} ({variance:+.1f})"),
        (f"Green: {green_pct:.0f}%", "of portfolio"),
        (f"Red: {red_count}", "projects"),
        (f"Total: {total_entries}", "entries"),
    ]

    col_w = 47
    for label, sub in kpis:
        pdf.cell(col_w, 12, label, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(148, 163, 184)
    for _, sub in kpis:
        pdf.cell(col_w, 6, sub, align='C')
    pdf.ln(10)

    # ── SCR boxes ──
    if scr:
        for key, color in [('situation', (59, 130, 246)), ('complications', (239, 68, 68)), ('resolution', (34, 197, 94))]:
            text = scr.get(key, '')
            if text:
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(*color)
                pdf.cell(0, 7, key.title(), ln=True)
                pdf.set_font('Helvetica', '', 9)
                pdf.set_text_color(226, 232, 240)
                pdf.multi_cell(0, 5, text)
                pdf.ln(3)

    # ── Chart image (if kaleido available) ──
    if chart_image_bytes:
        try:
            img_stream = io.BytesIO(chart_image_bytes)
            pdf.image(img_stream, x=10, w=190)
            pdf.ln(5)
        except Exception:
            pass

    # ── Key Insight ──
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 7, 'Key Insight', ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(226, 232, 240)
    pdf.multi_cell(0, 5, f"Weakest dimension: {worst_dim} (avg {worst_dim_score:.2f}/3.00)")
    pdf.ln(5)

    # ── Recommendations table ──
    if recommendations:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(59, 130, 246)
        pdf.cell(0, 7, 'Recommendations', ln=True)

        # Table header
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(224, 224, 224)
        pdf.set_fill_color(0, 30, 60)
        pdf.cell(15, 6, 'Pri', border=1, fill=True, align='C')
        pdf.cell(35, 6, 'Category', border=1, fill=True)
        pdf.cell(90, 6, 'Action', border=1, fill=True)
        pdf.cell(45, 6, 'Timeline', border=1, fill=True)
        pdf.ln()

        # Table rows
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(226, 232, 240)
        for rec in recommendations:
            pdf.cell(15, 6, rec.get('Priority', ''), border=1, align='C')
            pdf.cell(35, 6, rec.get('Category', ''), border=1)
            pdf.cell(90, 6, rec.get('Action', '')[:60], border=1)
            pdf.cell(45, 6, rec.get('Timeline', ''), border=1)
            pdf.ln()

    return bytes(pdf.output())
