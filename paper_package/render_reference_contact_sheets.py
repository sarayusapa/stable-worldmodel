#!/usr/bin/env python3
"""Render local paper PDFs into contact sheets for visual-design review."""

from pathlib import Path
import sys

import pypdfium2
from PIL import Image, ImageDraw


def render(pdf_path: Path, output: Path) -> None:
    document = pypdfium2.PdfDocument(pdf_path)
    thumbs = []
    for index in range(len(document)):
        image = document[index].render(scale=0.55).to_pil().convert("RGB")
        image.thumbnail((420, 560))
        thumbs.append((index, image.copy()))
    width = 420 * 4
    height = ((len(thumbs) + 3) // 4) * 590
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (index, image) in enumerate(thumbs):
        x = (position % 4) * 420
        y = (position // 4) * 590
        sheet.paste(image, (x, y + 25))
        draw.text((x + 5, y + 5), f"page {index + 1}", fill="black")
    sheet.save(output, quality=88)


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]))
