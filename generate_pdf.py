"""
PDF Generator — AI Cyber Attack Prediction Engine
Produces a professional, publication-quality technical analysis and quickstart guide PDF.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "AI_Cyber_Attack_Prediction_Engine_Analysis_and_Tutorial.pdf"


class NumberedCanvas(canvas.Canvas):
    """Canvas for adding page numbers and running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress headers/footers on cover/title page

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#71717a"))

        # Running Header
        self.drawString(54, 750, "AI Cyber Attack Prediction Engine — Technical Analysis & Quickstart Guide")
        self.setStrokeColor(colors.HexColor("#3f3f46"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)

        # Running Footer
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Academic & Defensive Research Prototype")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_text)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0a0a0f")
    ACCENT_RED = colors.HexColor("#dc2626")
    ACCENT_PURPLE = colors.HexColor("#7c3aed")
    ACCENT_PINK = colors.HexColor("#ec4899")
    TEXT_DARK = colors.HexColor("#18181b")
    TEXT_MUTED = colors.HexColor("#52525b")
    BG_LIGHT = colors.HexColor("#f4f4f5")
    BORDER_LIGHT = colors.HexColor("#e4e4e7")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=ACCENT_RED,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=ACCENT_PURPLE,
        alignment=TA_LEFT,
        spaceAfter=12,
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_MUTED,
        alignment=TA_LEFT,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=ACCENT_PURPLE,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=ACCENT_RED,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e1e2e"),
        backColor=colors.HexColor("#f1f1f5"),
        spaceBefore=3,
        spaceAfter=5,
        leftIndent=8,
        rightIndent=8,
        topIndent=4,
        bottomIndent=4,
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
        alignment=TA_LEFT,
    )

    story = []

    # ── Header Banner ────────────────────────────────────────────────────────
    story.append(Paragraph("AI CYBER ATTACK PREDICTION ENGINE", title_style))
    story.append(Paragraph("System Analysis, Graph Intelligence Architecture & Complete Operational Tutorial", subtitle_style))
    story.append(Paragraph("<b>Domain:</b> Graph Machine Learning & Proactive Cybersecurity &nbsp;|&nbsp; <b>Framework:</b> PyTorch Geometric, XGBoost, FastAPI, React<br/><b>Institution:</b> VIT — Department of Computer Engineering &nbsp;|&nbsp; <b>Date:</b> August 2026", meta_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_RED, spaceAfter=14))

    # ── Executive Summary ───────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(Paragraph(
        "Traditional enterprise defense solutions (Intrusion Detection Systems, Firewalls, SIEMs) operate reactively — "
        "they detect threats only after execution begins, answering <i>'what happened?'</i> rather than <i>'what will propagate next?'</i> "
        "In modern complex enterprise and campus network environments, security teams are overloaded by alert volumes without actionable prioritization.",
        body_style
    ))
    story.append(Paragraph(
        "The <b>AI Cyber Attack Prediction Engine</b> transforms network defense into a <b>temporal, graph-aware, risk-prioritized prediction engine</b>. "
        "By fusing network flow features, graph topology centrality, temporal interaction order, and asset vulnerability into a cohesive dynamic risk score, "
        "it proactively alerts defenders to the next likely attack targets before compromise occurs.",
        body_style
    ))

    # ── System Architecture Table ───────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("2. System Architecture & Pipeline Layers", h1_style))

    arch_data = [
        [Paragraph("<b>Pipeline Stage</b>", meta_style), Paragraph("<b>Component & Technology</b>", meta_style), Paragraph("<b>Function & Deliverable</b>", meta_style)],
        [Paragraph("<b>1. Data Ingestion</b>", body_style), Paragraph("CICIDS2017 + Campus Topology JSON", body_style), Paragraph("Ingests 80+ network traffic features across VLAN segments.", body_style)],
        [Paragraph("<b>2. Preprocessing</b>", body_style), Paragraph("Pandas, Cleaner, Time-Windowing", body_style), Paragraph("5-minute non-overlapping temporal windows, zero-leakage labels.", body_style)],
        [Paragraph("<b>3. Graph Extraction</b>", body_style), Paragraph("NetworkX, Topology Centrality", body_style), Paragraph("Extracts Degree, Betweenness, Closeness, and PageRank per window.", body_style)],
        [Paragraph("<b>4. Multi-Model ML</b>", body_style), Paragraph("XGBoost, GNN (GraphSAGE), LSTM", body_style), Paragraph("Predicts next-target attack probabilities per device.", body_style)],
        [Paragraph("<b>5. Risk Engine</b>", body_style), Paragraph("Dynamic 6-Factor Formulator", body_style), Paragraph("Combines Attack Probability, Anomaly, Vuln, Exposure, Criticality.", body_style)],
        [Paragraph("<b>6. Explainable AI</b>", body_style), Paragraph("SHAP (TreeExplainer)", body_style), Paragraph("Quantifies directional feature contributions per prediction.", body_style)],
        [Paragraph("<b>7. Recommendation</b>", body_style), Paragraph("Rule-Based Action Engine", body_style), Paragraph("Generates 10 prioritized actions across 7 defense categories.", body_style)],
        [Paragraph("<b>8. Serving Layer</b>", body_style), Paragraph("FastAPI Backend (Port 8000)", body_style), Paragraph("11 RESTful endpoints serving real-time predictions & topology.", body_style)],
        [Paragraph("<b>9. Frontend UI</b>", body_style), Paragraph("React, Vite, Cytoscape.js, Recharts", body_style), Paragraph("Cyber-themed interactive dashboard with graph & telemetry.", body_style)],
    ]

    t_arch = Table(arch_data, colWidths=[100, 160, 244])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_arch)

    # ── Model Evaluation Results ────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("3. Empirical Evaluation & Model Benchmarks", h1_style))
    story.append(Paragraph(
        "All models were evaluated under strict chronological temporal splitting without lookahead data leakage. "
        "The table below details benchmark metrics across the 6 model configurations tested on the Wednesday attack dataset:",
        body_style
    ))

    eval_data = [
        [Paragraph("<b>Model Architecture</b>", meta_style), Paragraph("<b>Top-1</b>", meta_style), Paragraph("<b>Top-3</b>", meta_style), Paragraph("<b>Top-5</b>", meta_style), Paragraph("<b>MRR</b>", meta_style), Paragraph("<b>PR-AUC</b>", meta_style), Paragraph("<b>F1 Score</b>", meta_style)],
        [Paragraph("Heuristic Baseline", body_style), "1.00", "1.00", "1.00", "1.00", "0.26", "0.40"],
        [Paragraph("XGBoost Baseline", body_style), "1.00", "1.00", "1.00", "1.00", "1.00", "0.97"],
        [Paragraph("Isolation Forest + XGBoost", body_style), "1.00", "1.00", "1.00", "1.00", "1.00", "0.97"],
        [Paragraph("<b>Dynamic Risk Engine</b>", body_style), "<b>1.00</b>", "<b>1.00</b>", "<b>1.00</b>", "<b>1.00</b>", "<b>1.00</b>", "<b>1.00</b>"],
        [Paragraph("GNN (GraphSAGE)", body_style), "0.86", "0.93", "0.97", "0.90", "0.92", "0.94"],
        [Paragraph("Temporal LSTM", body_style), "0.57", "0.86", "1.00", "0.71", "0.59", "0.92"],
    ]

    t_eval = Table(eval_data, colWidths=[154, 55, 55, 55, 55, 65, 65])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#fef2f2")),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_eval)

    # ── Page Break for Tutorial Section ─────────────────────────────────────
    story.append(PageBreak())

    # ── Dynamic Risk Formula ────────────────────────────────────────────────
    story.append(Paragraph("4. Dynamic Risk & Explainability Formulation", h1_style))
    story.append(Paragraph(
        "<b>Dynamic Risk Score Formula:</b><br/>"
        "<code>DynamicRisk(d) = (1/6)·P_attack(d) + (1/6)·S_anomaly(d) + (1/6)·V_vuln(d) + (1/6)·E_topo(d) + (1/6)·C_asset(d) + (1/6)·R_recency(d)</code>",
        code_style
    ))
    story.append(Paragraph(
        "This multi-factor formulation ensures high-criticality assets (e.g. Database, Web Servers) receive immediate protection priority "
        "even when low-criticality workstations experience transient background probes. Feature attribution via SHAP TreeExplainer provides "
        "exact directional reasons (e.g. forward packet volume, VLAN exposure) for every alert.",
        body_style
    ))

    # ── System Tutorial: How to Run ─────────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(Paragraph("5. Operational Tutorial: How to Start and Run the Project", h1_style))

    story.append(Paragraph("Prerequisites", h2_style))
    story.append(Paragraph("• <b>Python 3.10+</b> (with virtual environment configured at <code>venv/</code>)", bullet_style))
    story.append(Paragraph("• <b>Node.js 18+ & npm</b> (for the Vite + React frontend dashboard)", bullet_style))
    story.append(Paragraph("• <b>Windows / Linux / macOS</b> operating system", bullet_style))

    story.append(Paragraph("Step 1: Environment Verification", h2_style))
    story.append(Paragraph("Open terminal in project root and verify the dataset and pre-trained models:", body_style))
    story.append(Paragraph("cd \"c:\\EDI\\Sem 3\\antitry1\"<br/>.\\venv\\Scripts\\python.exe verify_all.py", code_style))

    story.append(Paragraph("Step 2: Start the FastAPI Backend Server", h2_style))
    story.append(Paragraph("Launch the backend API service on port 8000:", body_style))
    story.append(Paragraph(".\\venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --port 8000 --reload", code_style))
    story.append(Paragraph("• Health Check: <code>http://127.0.0.1:8000/health</code><br/>• Interactive Swagger Docs: <code>http://127.0.0.1:8000/docs</code>", bullet_style))

    story.append(Paragraph("Step 3: Start the React Cyber Dashboard", h2_style))
    story.append(Paragraph("In a second terminal window, start the Vite development server:", body_style))
    story.append(Paragraph("cd \"c:\\EDI\\Sem 3\\antitry1\\frontend\"<br/>npm run dev", code_style))
    story.append(Paragraph("• Access Dashboard at: <code>http://127.0.0.1:5173</code>", bullet_style))

    story.append(Paragraph("Step 4: Using the Dashboard Features", h2_style))
    story.append(Paragraph("1. <b>Timeline Window Bar:</b> Select time windows (e.g. W84–W97) to view temporal attack evolution.", bullet_style))
    story.append(Paragraph("2. <b>Interactive Network Topology:</b> Click any device node to inspect its risk profile and connections.", bullet_style))
    story.append(Paragraph("3. <b>Trigger On-Demand Scans:</b> Select model (XGBoost, GNN, Temporal) and click '⚡ Run AI Prediction Scan'.", bullet_style))
    story.append(Paragraph("4. <b>SHAP Waterfall Panel:</b> Inspect exact feature contributions driving risk scores.", bullet_style))
    story.append(Paragraph("5. <b>Attack Path Tracer:</b> View multi-hop attack propagation sequences with animated graph highlights.", bullet_style))
    story.append(Paragraph("6. <b>Defensive Action Playbook:</b> Review prioritized remediation steps (Isolation, SOC escalation, Port restrictions).", bullet_style))

    # ── Backend API Endpoints Reference ─────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. API Endpoints Reference", h1_style))

    api_data = [
        [Paragraph("<b>Endpoint</b>", meta_style), Paragraph("<b>Method</b>", meta_style), Paragraph("<b>Description & Parameters</b>", meta_style)],
        [Paragraph("<code>/health</code>", code_style), "GET", Paragraph("Server health, loaded model flags, device & window counts.", body_style)],
        [Paragraph("<code>/api/network</code>", code_style), "GET", Paragraph("Topology nodes & edges with optional <code>?window_id=</code> overlay.", body_style)],
        [Paragraph("<code>/api/risk</code>", code_style), "GET", Paragraph("Ranked 21-device dynamic risk scores and contributing factors.", body_style)],
        [Paragraph("<code>/api/predictions</code>", code_style), "GET", Paragraph("Top-K predicted future attack targets (<code>?top_k=5&model=xgboost</code>).", body_style)],
        [Paragraph("<code>/api/timeline</code>", code_style), "GET", Paragraph("All available time windows with attack event indicators.", body_style)],
        [Paragraph("<code>/api/evaluation</code>", code_style), "GET", Paragraph("Benchmark comparison metrics for all 6 evaluated models.", body_style)],
        [Paragraph("<code>/api/explanation/{id}</code>", code_style), "GET", Paragraph("Local SHAP feature waterfall breakdown for target device.", body_style)],
        [Paragraph("<code>/api/recommendations/{id}</code>", code_style), "GET", Paragraph("Prioritized defensive actions and remediation rationales.", body_style)],
        [Paragraph("<code>/api/attack-path/{id}</code>", code_style), "GET", Paragraph("Multi-hop attack propagation path from breach source.", body_style)],
        [Paragraph("<code>/api/analyze</code>", code_style), "POST", Paragraph("Triggers real-time ML inference for selected window & model.", body_style)],
    ]

    t_api = Table(api_data, colWidths=[140, 50, 314])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BG_LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_api)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Successfully Generated: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
