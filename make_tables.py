"""Renders each markdown table of draft.md as a PNG, so the tables survive pasting into Medium and WordPress.

    python3 make_figures.py

Reads the tables straight from draft.md, so the images always match the text. Words are never split across lines.
"""
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
NAVY, INK, ZEBRA, RULE = (16, 42, 67), (28, 36, 48), (241, 245, 249), (203, 213, 225)
S = 2                                   # draw at double size for a sharp picture
NAMES = ["table1_haiku_accuracy", "table2_hand_off_vs_direct", "table3_qwen_accuracy", "table4_translation_runs", "table5_result_to_action"]


def tables(md):
    out, cur = [], []
    for line in md.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def cells(line):
    return [re.sub(r"`", "", c.strip()) for c in line.strip().strip("|").split("|")]


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def render(rows, aligns, path):
    f, fb = ImageFont.truetype(FONT, 17 * S), ImageFont.truetype(BOLD, 17 * S)
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    n, pad, total = len(rows[0]), 14 * S, 1500 * S
    want = []
    for j in range(n):
        longest = max(d.textlength(r[j], font=fb if i == 0 else f) for i, r in enumerate(rows))
        want.append(min(longest, 520 * S) + 2 * pad)
    scale = min(1.0, total / sum(want))
    widths = [w * scale for w in want]
    if sum(widths) < total:                       # give spare width to the widest text column
        extra = (total - sum(widths)) / n
        widths = [w + extra for w in widths]
    wrapped = [[wrap(d, c, fb if i == 0 else f, widths[j] - 2 * pad) for j, c in enumerate(r)] for i, r in enumerate(rows)]
    line_h = 24 * S
    heights = [max(len(c) for c in r) * line_h + 2 * 10 * S for r in wrapped]
    img = Image.new("RGB", (int(sum(widths)), sum(heights)), "white")
    dr = ImageDraw.Draw(img)
    y = 0
    for i, r in enumerate(wrapped):
        h = heights[i]
        dr.rectangle([0, y, img.width, y + h], fill=NAVY if i == 0 else (ZEBRA if i % 2 == 0 else "white"))
        x = 0
        for j, c in enumerate(r):
            for k, ln in enumerate(c):
                fnt = fb if i == 0 else f
                tw = dr.textlength(ln, font=fnt)
                tx = x + pad if aligns[j] == "l" else x + widths[j] - pad - tw
                dr.text((tx, y + 10 * S + k * line_h), ln, font=fnt, fill="white" if i == 0 else INK)
            x += widths[j]
        dr.line([0, y + h, img.width, y + h], fill=RULE, width=S)
        y += h
    img = img.resize((img.width // S, img.height // S), Image.LANCZOS)
    img.save(path)
    return img.size


def main():
    ts = tables((HERE / "draft.md").read_text())
    assert len(ts) == len(NAMES), f"expected {len(NAMES)} tables, found {len(ts)}"
    for name, t in zip(NAMES, ts):
        aligns = ["r" if c.strip().endswith(":") else "l" for c in t[1].strip().strip("|").split("|")]
        rows = [cells(t[0])] + [cells(x) for x in t[2:]]
        size = render(rows, aligns, HERE / "figures" / f"{name}.png")
        print(name, size)


if __name__ == "__main__":
    main()
