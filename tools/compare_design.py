"""Combine real reference and browser captures for a visible QA comparison."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("implementation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    images = [Image.open(path).convert("RGB") for path in (args.reference, args.implementation)]
    width = 1487
    images = [im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS) for im in images]
    canvas = Image.new("RGB", (width * 2 + 24, max(im.height for im in images) + 44), "#15191F")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    for i, (im, label) in enumerate(zip(images, ("Selected reference", "Live implementation — actual model output"), strict=True)):
        x = i * (width + 24)
        draw.text((x + 12, 12), label, font=font, fill="#F5F7FA")
        canvas.paste(im, (x, 44))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    # The rail contains the two requested fidelity surfaces: body map and feed.
    crops = [im.crop((0, 0, 280, min(im.height, 1058))) for im in images]
    detail = Image.new("RGB", (584, max(im.height for im in crops) + 44), "#15191F")
    for i, im in enumerate(crops):
        detail.paste(im, (i * 304, 44))
    ImageDraw.Draw(detail).text((8, 12), "Reference / implementation — fighter rail", font=font, fill="#F5F7FA")
    detail.save(args.output.with_name(args.output.stem + "-detail.png"))
    print(args.output)


if __name__ == "__main__":
    main()
