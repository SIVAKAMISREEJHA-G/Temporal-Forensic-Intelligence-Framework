"""
report_generator.py — LLM-powered forensic report + PDF generation.
Set LLM_PROVIDER=anthropic (default) or openai in .env / environment.
"""
import os, json, sys, re
from datetime import datetime

# ── PDF generation ────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, Image as RLImage,
                                 PageBreak)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

REPORT_PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORT_PDF_DIR, exist_ok=True)

CLASSES   = ["Abuse","Car Accident","Explosion","Fighting","Normal","Riot","Shooting"]
SEVERITY  = {"Normal":"LOW","Car Accident":"MEDIUM","Abuse":"HIGH",
             "Fighting":"HIGH","Riot":"HIGH","Shooting":"CRITICAL","Explosion":"CRITICAL"}
SEVERITY_COLOR = {"LOW":"#27ae60","MEDIUM":"#f39c12","HIGH":"#e67e22","CRITICAL":"#e74c3c"}


# ─── LLM call ─────────────────────────────────────────────────────────────────
def _call_llm(prompt: str) -> dict:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    try:
        if provider == "openai":
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are an expert forensic analyst."},
                          {"role":"user","content":prompt}],
                response_format={"type":"json_object"},
                temperature=0.3, max_tokens=2000,
            )
            raw = resp.choices[0].message.content
        else:  # anthropic
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
            resp = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=[{"role":"user","content":prompt}],
            )
            raw = resp.content[0].text

        # extract JSON block if wrapped in ```json ... ```
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        raw = m.group(1) if m else raw
        return json.loads(raw)
    except Exception as e:
        print(f"LLM call failed ({provider}): {e}")
        return None


def _build_prompt(video_id, filename, clf, temporal, meta):
    cls  = clf["predicted_class"]
    conf = clf["confidence"]
    per  = clf["per_class_scores"]
    segs = temporal["segments"]
    kfs  = temporal["keyframes"]
    dur  = meta["duration"]
    sev  = SEVERITY.get(cls, "MEDIUM")

    timeline_text = "\n".join(
        f"  • {s['label']} ({s['time_start']}s–{s['time_end']}s): "
        f"activity_level={s['activity_level']}, avg_attention={s['avg_attention']}"
        for s in segs
    )
    scores_text = ", ".join(f"{k}={v:.2%}" for k,v in per.items())

    prompt = f"""You are a professional forensic video analyst. Analyse the following surveillance video data and produce a complete forensic report in valid JSON.

VIDEO METADATA:
- File: {filename}
- Case ID: TFIF-{video_id:06d}
- Duration: {dur}s
- Analysed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

CLASSIFICATION RESULT:
- Predicted Class: {cls}
- Confidence: {conf:.2%}
- Severity: {sev}
- All Class Scores: {scores_text}

TEMPORAL TIMELINE:
{timeline_text}

KEY EVIDENCE FRAMES: {len(kfs)} salient frames identified at timestamps: {', '.join(str(k['timestamp_sec'])+'s' for k in kfs)}

Produce a JSON object with EXACTLY these keys (do NOT add extra keys):
{{
  "executive_summary": "<2-3 sentence summary of the incident or scene>",
  "incident_description": "<detailed paragraph>",
  "detected_activities": ["<activity 1>", "<activity 2>", ...],
  "threat_assessment": "<formal threat assessment paragraph>",
  "chronological_summary": "<paragraph narrating events in order>",
  "investigator_observations": "<paragraph of technical observations>",
  "recommended_actions": ["<action 1>", "<action 2>", ...],
  "final_conclusion": "<concluding paragraph>"
}}

Write in formal forensic report tone. Be specific and professional. Even for Normal videos, describe the observed routine activity in detail."""

    return prompt


def _fallback_report(video_id, filename, cls, conf, segs):
    """Generate a deterministic report when LLM is unavailable."""
    sev = SEVERITY.get(cls,"MEDIUM")
    timeline = "; ".join(f"{s['label']} ({s['time_start']:.1f}s–{s['time_end']:.1f}s)" for s in segs)
    return {
        "executive_summary": f"Automated analysis classified this video as '{cls}' with {conf:.1%} confidence (severity: {sev}). LLM narrative unavailable — see raw metrics below.",
        "incident_description": f"The AI system detected '{cls}' activity throughout the footage. Confidence={conf:.1%}. Timeline: {timeline}.",
        "detected_activities": [cls, f"Severity: {sev}"],
        "threat_assessment": f"Threat level assessed as {sev} based on classification confidence of {conf:.1%}.",
        "chronological_summary": timeline,
        "investigator_observations": "Automated system observation — LLM provider unavailable or API key not set.",
        "recommended_actions": ["Review raw footage manually", "Cross-reference with other sensor data"],
        "final_conclusion": f"The incident is classified as '{cls}'. Human review recommended.",
    }


# ─── PDF builder ──────────────────────────────────────────────────────────────
def _build_pdf(video_id, filename, cls, conf, sev, report_json, temporal, meta):
    pdf_path = os.path.join(REPORT_PDF_DIR, f"report_{video_id:06d}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("Title2", fontName="Helvetica-Bold", fontSize=18,
                                  textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("Sub", fontName="Helvetica", fontSize=10,
                                  textColor=colors.HexColor("#555"), alignment=TA_CENTER, spaceAfter=12)
    h2_style    = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13,
                                  textColor=colors.HexColor("#1a1a2e"), spaceBefore=14, spaceAfter=4)
    body_style  = ParagraphStyle("Body", fontName="Helvetica", fontSize=10,
                                  leading=15, textColor=colors.HexColor("#222"), alignment=TA_JUSTIFY)
    bullet_style= ParagraphStyle("Bullet", fontName="Helvetica", fontSize=10, leading=14,
                                  leftIndent=16, textColor=colors.HexColor("#333"))

    sev_color_map = {"LOW":"#27ae60","MEDIUM":"#f39c12","HIGH":"#e67e22","CRITICAL":"#c0392b"}
    sev_hex = sev_color_map.get(sev,"#999")

    story = []

    # Header
    story.append(Paragraph("TFIF — TEMPORAL FORENSIC INTELLIGENCE FRAMEWORK", title_style))
    story.append(Paragraph("Forensic Video Analysis Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 10))

    # Meta table
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    meta_data = [
        ["Case ID", f"TFIF-{video_id:06d}", "File Name", filename],
        ["Date", now,                        "Duration",  f"{meta['duration']}s"],
        ["Resolution", meta.get('resolution','N/A'), "FPS", str(meta.get('fps','N/A'))],
        ["Classification", cls,              "Confidence", f"{conf:.1%}"],
        ["Severity", sev, "", ""],
    ]
    meta_table = Table(meta_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 4.5*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#1a1a2e")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0,0), (0,-1), colors.white),
        ("TEXTCOLOR",  (2,0), (2,-1), colors.white),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (1,0), (-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#ddd")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Severity badge
    sev_table = Table([[f"⚠  SEVERITY: {sev}"]], colWidths=["100%"])
    sev_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor(sev_hex)),
        ("TEXTCOLOR",  (0,0), (-1,-1), colors.white),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 11),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), 4),
    ]))
    story.append(sev_table)
    story.append(Spacer(1, 14))

    def section(title, content):
        story.append(Paragraph(title, h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccc")))
        story.append(Spacer(1, 4))
        if isinstance(content, list):
            for item in content:
                story.append(Paragraph(f"• {item}", bullet_style))
        else:
            story.append(Paragraph(str(content).replace("\n","<br/>"), body_style))
        story.append(Spacer(1, 8))

    section("Executive Summary",        report_json.get("executive_summary",""))
    section("Incident Description",     report_json.get("incident_description",""))
    section("Detected Activities",      report_json.get("detected_activities",[]))
    section("Threat Assessment",        report_json.get("threat_assessment",""))
    section("Chronological Event Summary", report_json.get("chronological_summary",""))
    section("Investigator Observations",  report_json.get("investigator_observations",""))
    section("Recommended Actions",      report_json.get("recommended_actions",[]))
    section("Final Conclusion",         report_json.get("final_conclusion",""))

    # Timeline table
    story.append(Paragraph("Temporal Event Timeline", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccc")))
    story.append(Spacer(1, 4))
    tl_header = [["Segment", "Label", "Time Window", "Activity Level", "Attention"]]
    tl_rows   = [[str(s["segment"]), s["label"], f"{s['time_start']}s–{s['time_end']}s",
                  s["activity_level"], f"{s['avg_attention']:.4f}"]
                 for s in temporal["segments"]]
    tl_table = Table(tl_header + tl_rows, colWidths=[2*cm, 5*cm, 4*cm, 4*cm, 3*cm])
    tl_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#ddd")),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(tl_table)
    story.append(Spacer(1, 14))

    # Key evidence frames
    kfs = temporal.get("keyframes", [])
    if kfs:
        story.append(Paragraph("Key Evidence Frames", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccc")))
        story.append(Spacer(1, 6))
        img_cells = []
        for kf in kfs[:4]:
            p = kf.get("path","")
            if p and os.path.exists(p):
                try:
                    img = RLImage(p, width=4*cm, height=3*cm)
                    label = f"t={kf['timestamp_sec']}s\nattn={kf['attention']:.4f}"
                    img_cells.append([img, Paragraph(label, ParagraphStyle("tiny", fontSize=7))])
                except Exception:
                    img_cells.append(["[Frame unavailable]", ""])
        if img_cells:
            kf_table = Table(img_cells, colWidths=[4.2*cm, 2.3*cm] * min(2, len(img_cells)))
            story.append(kf_table)

    doc.build(story)
    return pdf_path


# ═══════════════════════════════════════════════════════════════════════════════
class ReportGenerationAgent:
    def generate(self, video_id, video_path, clf, temporal, meta):
        filename  = os.path.basename(video_path)
        cls       = clf["predicted_class"]
        conf      = clf["confidence"]
        sev       = SEVERITY.get(cls,"MEDIUM")

        prompt = _build_prompt(video_id, filename, clf, temporal, meta)
        report_json = _call_llm(prompt)
        if report_json is None:
            report_json = _fallback_report(video_id, filename, cls, conf, temporal["segments"])

        pdf_path = _build_pdf(video_id, filename, cls, conf, sev, report_json, temporal, meta)
        return {"report_json": report_json, "pdf_path": pdf_path}
