import json

import numpy as np
from PIL import ImageColor, ImageDraw, ImageFont


def _get_coco_classes(json_file: str = "models/instances_val2017.json"):
    with open(json_file, "r") as COCO:
        js = json.loads(COCO.read())
    return {cat["id"]: cat["name"] for cat in js["categories"]}


coco_classes = _get_coco_classes()


def draw_detection(draw: ImageDraw, d, c):
    """Draw box and label for 1 detection."""
    width, height = draw.im.size
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = max(0, np.floor(top + 0.5).astype("int32"))
    left = max(0, np.floor(left + 0.5).astype("int32"))
    bottom = min(height, np.floor(bottom + 0.5).astype("int32"))
    right = min(width, np.floor(right + 0.5).astype("int32"))
    label = coco_classes[c]
    label_size = draw.textsize(label, stroke_width=10)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("red")
    thickness = 10
    draw.rectangle(
        [
            left + thickness,
            top + thickness,
            right - thickness,
            bottom - thickness,
        ],
        outline=color,
    )
    # font = ImageFont.truetype(
    #     "/Users/aminedirhoussi/Downloads/OpenSans-Bold.ttf", size=20
    # )
    draw.text(text_origin, label, fill=color)  #  font=font)
