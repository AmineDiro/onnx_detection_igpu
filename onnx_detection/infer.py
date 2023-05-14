import sys

import numpy as np
import onnxruntime as rt
from PIL import Image, ImageDraw
from utils import draw_detection


def process_image(path: str):
    img = Image.open(path)
    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)
    return img, img_data


if __name__ == "__main__":
    IMG_PATH = "data/pizza.jpeg"
    MODEL = "models/ssd_mobilenet_v1_10.onnx"

    img, img_data = process_image(IMG_PATH)

    sess = rt.InferenceSession(MODEL)

    # we want the outputs in this order
    outputs = [
        "num_detections:0",
        "detection_boxes:0",
        "detection_scores:0",
        "detection_classes:0",
    ]

    result = sess.run(outputs, {"image_tensor:0": img_data})
    (
        num_detections,
        detection_boxes,
        detection_scores,
        detection_classes,
    ) = result

    batch_size = num_detections.shape[0]
    draw = ImageDraw.Draw(img)
    for batch in range(0, batch_size):
        for detection in range(0, int(num_detections[batch])):
            c = detection_classes[batch][detection]
            d = detection_boxes[batch][detection]

            draw_detection(draw, d, c)

    img.save(f"{IMG_PATH}_result.png")
