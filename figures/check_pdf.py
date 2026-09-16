#!/usr/bin/env python3
"""WHITEPAPER.md vs WHITEPAPER.pdf parity gate.

Checks that every prose line, heading, bullet and table cell of the markdown
source appears verbatim in the PDF's extracted text (page footers filtered
out geometrically -- they extract interleaved with body text), plus basic
structural sanity: links, images, bounds, empty pages.

Exit 0 = PASS. Run after every make_pdf.py rebuild.
"""
import os, re, sys
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOOT = re.compile(r"(FSR4 on a Phone NPU\s*\d*|\d+)")

doc = pymupdf.open(os.path.join(ROOT, "WHITEPAPER.pdf"))
keep, links, imgs, oob, empty = [], 0, 0, [], []
for i, p in enumerate(doc):
    H = p.rect.height
    blocks = p.get_text("blocks")
    keep.append(" ".join(b[4] for b in blocks
                         if not (b[1] > H - 45 and b[3] < H - 15 and FOOT.fullmatch(b[4].strip()))))
    links += len(p.get_links())
    imgs += len(p.get_images())
    if not p.get_text().strip():
        empty.append(i + 1)
    W = p.rect.width
    for b in blocks:
        if b[3] > H + 1 or b[1] < -1 or b[2] > W + 1 or b[0] < -1:
            oob.append(i + 1)
pdf_n = re.sub(r"\s+", " ", " ".join(keep)).strip()

def strip_inline(s):
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return s

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

missing = []
for ln, raw in enumerate(open(os.path.join(ROOT, "WHITEPAPER.md"), encoding="utf-8").read().splitlines(), 1):
    s = raw.strip()
    if not s or s.startswith("---") or s.startswith("!["):
        continue
    if s.startswith("|") and set(s.replace("|", "").replace(" ", "")) <= set("-"):
        continue
    s = strip_inline(s)
    if s.startswith("#"):
        s = s.lstrip("# ")
    if s.startswith("|"):
        for c in s.strip("|").split("|"):
            c = norm(c)
            if len(c.split()) >= 2 and c not in pdf_n:
                missing.append((ln, "cell", c[:70]))
        continue
    if s.startswith("- "):
        s = s[2:]
    if len(s.split()) < 4:
        continue
    if norm(s) not in pdf_n:
        missing.append((ln, "line", norm(s)[:90]))

ok = True
if missing:
    ok = False
    print(f"FAIL: {len(missing)} markdown segment(s) missing from PDF text:")
    for m in missing:
        print(f"  line {m[0]} {m[1]}: {m[2]}")
if oob:
    ok = False
    print(f"FAIL: out-of-bounds text blocks on pages {sorted(set(oob))}")
if empty:
    ok = False
    print(f"FAIL: empty pages {empty}")
if imgs < 4:
    ok = False
    print(f"FAIL: expected at least 4 embedded figures, found {imgs}")
if links < 5:
    ok = False
    print(f"FAIL: expected at least 5 live links, found {links}")

print(f"pages: {len(doc)} | words: {len(pdf_n.split())} | links: {links} | images: {imgs}")
print("PDF PARITY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
