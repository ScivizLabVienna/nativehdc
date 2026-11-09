# -*- coding: utf-8 -*-
"""Build paper #2 PDF from docs/paper2-draft.md (markdown stays canonical). SYSTEM python."""
import os, re
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether
from reportlab.lib import colors

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "..", "docs", "paper2-draft.md")
FIGS = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "paper2-draft-v0.1.pdf")

import matplotlib
FD = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for name, f in [("Body", "DejaVuSerif.ttf"), ("Body-B", "DejaVuSerif-Bold.ttf"),
                ("Body-I", "DejaVuSerif-Italic.ttf"), ("Sans", "DejaVuSans.ttf"),
                ("Sans-B", "DejaVuSans-Bold.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(FD, f)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-B")

TITLE = ParagraphStyle("t", fontName="Sans-B", fontSize=16.5, leading=20.5, spaceAfter=4)
SUB = ParagraphStyle("s", fontName="Sans", fontSize=8.6, leading=11.5, textColor=colors.HexColor("#666677"), spaceAfter=12)
H1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=12.3, leading=15, spaceBefore=13, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
H2 = ParagraphStyle("h2", parent=H1, fontSize=10.8, spaceBefore=10)
BODY = ParagraphStyle("b", fontName="Body", fontSize=9.8, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=7)
CAP = ParagraphStyle("c", fontName="Body-I", fontSize=8.4, leading=10.8, textColor=colors.HexColor("#444444"), spaceAfter=9, spaceBefore=2)

# figures land at the END of the mapped section (flushed when the next heading arrives)
FIG_AFTER_SEC = {"3.1": ["fig1_x1.png", "fig3_x4.png"], "3.2": ["fig2_x2.png"]}
CAPS = {"fig1_x1.png": "Figure 1: X1 - chart arms hold 1.000 across D while flat bundles climb the interference knee (3 seeds per point).",
        "fig2_x2.png": "Figure 2: X2 - relation-pool charts degrade continuously with n_rel (key-aliasing continuum); flat stays below 0.10 regardless.",
        "fig3_x4.png": "Figure 3: X4 - depth-3 chains: chart arms 1.000 vs flat ~0.025."}

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", t)
    return t.replace("`", "")

def fig_block(name):
    from reportlab.lib.utils import ImageReader
    iw0, ih0 = ImageReader(os.path.join(FIGS, name)).getSize()
    w = 14.2 * cm
    story_items = [Spacer(1, 4), Image(os.path.join(FIGS, name), width=w, height=w * ih0 / iw0),
                   Paragraph(CAPS[name], CAP)]
    return KeepTogether(story_items)

lines = open(MD, encoding="utf-8").read().split("\n")
story, i, cur = [], 0, None      # cur = section id whose figures are pending ("3.1", "3.2", ...)
def flush():
    global cur
    if cur in FIG_AFTER_SEC:
        for f in FIG_AFTER_SEC[cur]:
            story.append(fig_block(f))
    cur = None

while i < len(lines):
    ln = lines[i].strip(); i += 1
    if not ln or ln == "---":
        continue
    m = re.match(r"^(#{1,3})\s+(.*)$", ln)
    if m:
        flush()
        lvl, txt = len(m.group(1)), inline(m.group(2))
        if lvl == 1:
            story.append(Paragraph(txt, TITLE))
            story.append(Paragraph("Draft v0.1 for operator review &middot; ScivizLab Vienna &middot; 2026-11-08 &middot; "
                                   "numbers transcribed from frozen pre-registered ledgers G10-G13", SUB))
            cur = None
        else:
            (H1 if lvl == 2 else H2) and story.append(Paragraph(txt, H1 if lvl == 2 else H2))
            m2 = re.match(r"^(\d\.\d)\s", m.group(2))
            cur = m2.group(1) if m2 else None
        continue
    while i < len(lines) and lines[i].strip() and not re.match(r"^#{1,3}\s", lines[i]) and lines[i].strip() != "---":
        ln += " " + lines[i].strip(); i += 1
    story.append(Paragraph(inline(ln), BODY))
flush()

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                  topMargin=2.0 * cm, bottomMargin=2.0 * cm, title="Partition over Gluing (draft v0.1)").build(story)

n = open(OUT, "rb").read()
print("PDF", len(n), "bytes | header ok:", n[:5] == b"%PDF-", "| pages ~", n.count(b"/Type /Page ") + n.count(b"/Type/Page "))
