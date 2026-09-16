#!/usr/bin/env python3
# WHITEPAPER.md -> WHITEPAPER.pdf via ReportLab. Markdown -> HTML -> Flowables.
import os, re, sys
import markdown as md_lib
import xml.etree.ElementTree as ET
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "WHITEPAPER.md")
OUT = os.path.join(ROOT, "WHITEPAPER.pdf")

# ---- local fonts (host-first) ------------------------------------------------
FDIR = r"C:\Windows\Fonts"
def reg(name, candidates):
    for c in candidates:
        p = os.path.join(FDIR, c)
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont(name, p))
            return name
    return None

body_f  = reg("Georgia", ["georgia.ttf"]) or "Times-Roman"
ital_f  = reg("Georgia-Italic", ["georgiai.ttf"]) or "Times-Italic"
bold_f  = reg("Georgia-Bold", ["georgiab.ttf"]) or "Times-Bold"
head_f  = reg("Segoe-UI", ["segoeui.ttf"]) or "Helvetica"
headb_f = reg("Segoe-UI-Bold", ["segoeuib.ttf"]) or "Helvetica-Bold"
code_f  = reg("Consolas", ["consola.ttf"]) or "Courier"

INK    = colors.HexColor("#1a1a1a")
ACCENT = colors.HexColor("#2b5f8a")
MUTE   = colors.HexColor("#666666")
RULE   = colors.HexColor("#c9c9c9")
TBG    = colors.HexColor("#f2f5f8")

S = {
    "title": ParagraphStyle("t",  fontName=headb_f, fontSize=25, leading=30,
                            textColor=INK, spaceAfter=4*mm),
    "sub":   ParagraphStyle("s",  fontName=ital_f, fontSize=12, leading=16,
                            textColor=MUTE, spaceAfter=8*mm),
    "h2":    ParagraphStyle("h2", fontName=headb_f, fontSize=15.5, leading=19,
                            textColor=INK, spaceBefore=9*mm, spaceAfter=3.5*mm),
    "h3":    ParagraphStyle("h3", fontName=headb_f, fontSize=12, leading=15.5,
                            textColor=ACCENT, spaceBefore=6*mm, spaceAfter=2.5*mm),
    "body":  ParagraphStyle("b",  fontName=body_f, fontSize=10.5, leading=15.2,
                            textColor=INK, spaceAfter=3.2*mm, alignment=TA_LEFT),
    "bullet":ParagraphStyle("bl", fontName=body_f, fontSize=10.5, leading=15.2,
                            textColor=INK, spaceAfter=2.4*mm, leftIndent=6*mm,
                            bulletIndent=2*mm),
    "caption":ParagraphStyle("c", fontName=ital_f, fontSize=9.3, leading=12.6,
                            textColor=MUTE, spaceBefore=1.5*mm, spaceAfter=5*mm),
    "cell":  ParagraphStyle("cl", fontName=body_f, fontSize=9.4, leading=12.4,
                            textColor=INK),
    "cellh": ParagraphStyle("ch", fontName=headb_f, fontSize=9.4, leading=12.4,
                            textColor=INK),
}

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline_to_rl(el, style):
    """Convert inline HTML (ET element or string) to ReportLab mini-markup."""
    if isinstance(el, str):
        return esc(el)
    tag = el.tag.split("}")[-1]
    parts = []
    for c in el:
        parts.append(inline_to_rl(c, style) if isinstance(c, ET.Element) else esc(c))
        if c.tail:
            parts.append(esc(c.tail))
    if el.text:
        parts.insert(0, esc(el.text))
    inner = "".join(parts)
    if tag == "strong" or tag == "b":
        return f"<font name=\"{bold_f}\">{inner}</font>"
    if tag == "em" or tag == "i":
        return f"<font name=\"{ital_f}\">{inner}</font>"
    if tag == "code":
        return f"<font name=\"{code_f}\" size=\"9.6\">{inner}</font>"
    if tag == "a":
        href = el.get("href", "")
        return f"<link href=\"{esc(href)}\" color=\"#2b5f8a\"><u>{inner}</u></link>"
    return inner

def para_text(p_el):
    parts = []
    for c in p_el:
        parts.append(inline_to_rl(c, None) if isinstance(c, ET.Element) else esc(c))
        if c.tail:
            parts.append(esc(c.tail))
    if p_el.text:
        parts.insert(0, esc(p_el.text))
    return "".join(parts)

def strip_ns(tag):
    return tag.split("}")[-1]

def build_table(tbl):
    rows = []
    # find all tr regardless of namespace
    trs = [e for e in tbl.iter() if strip_ns(e.tag) == "tr"]
    for tr in trs:
        cells = []
        for td in tr:
            t = strip_ns(td.tag)
            if t in ("td", "th"):
                cells.append(Paragraph(para_text(td), S["cellh"] if t == "th" else S["cell"]))
        if cells:
            rows.append(cells)
    if not rows:
        return None
    ncols = max(len(r) for r in rows)
    avail = A4[0] - 30*mm
    widths = [avail / ncols] * ncols
    t = Table(rows, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TBG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafb")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

def build_image(img):
    src = img.get("src", "")
    p = os.path.join(ROOT, src)
    if not os.path.exists(p):
        return None
    avail = A4[0] - 30*mm
    with PILImage.open(p) as im:
        w, h = im.size
    if w > avail:
        w, h = avail, h * avail / w
    return Image(p, width=w, height=h)

# ---- parse markdown to HTML ---------------------------------------------------
text = open(SRC, encoding="utf-8").read()
html = md_lib.markdown(text, extensions=["tables", "sane_lists"])
html = f"<root>{html}</root>"
root = ET.fromstring(html)

story = []
first_h1 = True
pending_caption = None
for el in root:
    tag = strip_ns(el.tag)
    if tag == "h1":
        story.append(Paragraph(para_text(el), S["title"]))
    elif tag == "p":
        # image-only paragraph?
        imgs = [e for e in el.iter() if strip_ns(e.tag) == "img"]
        if imgs and not (el.text or "").strip() and len(imgs) == 1 and not any(
                strip_ns(c.tag) not in ("img",) for c in el):
            im = build_image(imgs[0])
            if im:
                story.append(Spacer(1, 4*mm))
                story.append(im)
            continue
        txt = para_text(el)
        # caption = italic-only paragraph right after an image
        if el.text and not el.text.strip() and len(el) == 1 and strip_ns(el[0].tag) == "em":
            story.append(Paragraph(para_text(el[0]), S["caption"]))
        else:
            story.append(Paragraph(txt, S["body"]))
    elif tag == "h2":
        story.append(Paragraph(para_text(el), S["h2"]))
    elif tag in ("h3", "h4"):
        story.append(Paragraph(para_text(el), S["h3"]))
    elif tag == "hr":
        story.append(Spacer(1, 2*mm))
        story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
        story.append(Spacer(1, 2*mm))
    elif tag == "table":
        t = build_table(el)
        if t:
            story.append(Spacer(1, 1.5*mm))
            story.append(t)
            story.append(Spacer(1, 4*mm))
    elif tag in ("ul", "ol"):
        for li in el:
            if strip_ns(li.tag) != "li":
                continue
            story.append(Paragraph(para_text(li), S["bullet"], bulletText="\u2022"))
        story.append(Spacer(1, 1.5*mm))

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(head_f, 8)
    canvas.setFillColor(MUTE)
    canvas.drawString(15*mm, 9*mm, "FSR4 on a Phone NPU")
    canvas.drawRightString(A4[0] - 15*mm, 9*mm, str(canvas.getPageNumber()))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(15*mm, 12*mm, A4[0] - 15*mm, 12*mm)
    canvas.restoreState()

doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=15*mm, rightMargin=15*mm, topMargin=16*mm, bottomMargin=18*mm,
    title="FSR4 on a Phone NPU: The Retroid Pocket 6 DLSS Heist",
    author="puzzled-pancake",
    subject="Running AMD FSR4 W8A8 on the Hexagon NPU of a Retroid Pocket 6 via DLSS call interception",
    creator="puzzled-pancake",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
