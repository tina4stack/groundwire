#!/usr/bin/env python3
"""Generate the groundwire brand icon -> groundwire/assets/groundwire.ico + tray PNGs.

The mark: a warm terracotta spiral (memory coiling inward -> retrieval spiralling
back out) inside a dotted ring on a cream disc with a dark rim. Hand-drawn,
recognisable at 16px. Run once; commit the assets. Pure Pillow."""
import math
import os
from PIL import Image, ImageDraw

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "groundwire", "assets")
CREAM = (244, 240, 230)     # paper disc
RIM = (38, 33, 29)          # dark outer ring + dots
RUST = (193, 88, 49)        # the spiral
GREY = (150, 145, 138)      # paused spiral


def render(size: int, paused: bool = False) -> Image.Image:
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2

    # cream disc
    d.ellipse([12, 12, S - 12, S - 12], fill=CREAM)

    # dotted inner ring
    dot_r = c - 34
    ndots = 60
    for i in range(ndots):
        a = 2 * math.pi * i / ndots
        x, y = c + dot_r * math.cos(a), c + dot_r * math.sin(a)
        d.ellipse([x - 2.6, y - 2.6, x + 2.6, y + 2.6], fill=RIM)

    # terracotta spiral: filled dots along an Archimedean path give a smooth
    # thick stroke with round caps. ~2.6 turns, starting at the centre.
    spiral = GREY if paused else RUST
    turns, steps, max_r, w = 2.6, 480, 74.0, 15.0
    for i in range(steps + 1):
        t = i / steps
        theta = turns * 2 * math.pi * t
        r = max_r * t
        x, y = c + r * math.cos(theta), c + r * math.sin(theta)
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=spiral)
    # round the very centre so the spiral starts as a solid dot, not a point
    d.ellipse([c - w / 2, c - w / 2, c + w / 2, c + w / 2], fill=spiral)

    # crisp dark rim on top of everything
    d.ellipse([12, 12, S - 12, S - 12], outline=RIM, width=9)

    if size != S:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def main():
    os.makedirs(ASSETS, exist_ok=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [render(s) for s in sizes]
    ico = os.path.join(ASSETS, "groundwire.ico")
    frames[-1].save(ico, sizes=[(s, s) for s in sizes],
                    append_images=frames[:-1])
    render(64).save(os.path.join(ASSETS, "groundwire-64.png"))
    render(256).save(os.path.join(ASSETS, "groundwire.png"))
    render(64, paused=True).save(os.path.join(ASSETS, "groundwire-paused-64.png"))
    print(f"wrote {ico} and PNGs ({', '.join(map(str, sizes))})")


if __name__ == "__main__":
    main()
