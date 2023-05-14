import argparse
import time

import cv2
import numpy as np
import onnxruntime as rt
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

from onnx_detection.utils import coco_classes, draw_detection

_ROW_SIZE = 50  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 2.4
_FONT_THICKNESS = 2
_FPS_AVERAGE_FRAME_COUNT = 10


def run(model, rtsp_username, rtsp_password, width, height):
    cam_url = f"rtsp://{rtsp_username}:{rtsp_password}@192.168.1.88:554/ch4"
    cap = cv2.VideoCapture(cam_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Loading model
    sess_opt = rt.SessionOptions()
    sess_opt.intra_op_num_threads = 6
    sess = rt.InferenceSession(model, sess_options=sess_opt)
    outputs = [
        "num_detections:0",
        "detection_boxes:0",
        "detection_scores:0",
        "detection_classes:0",
    ]

    while cap.isOpened():
        ret, image = cap.read()

        image = np.expand_dims(image.astype(np.uint8), axis=0)
        counter += 1

        # Running inference on ONNX
        (
            num_detections,
            detection_boxes,
            detection_scores,
            detection_classes,
        ) = sess.run(outputs, {"image_tensor:0": image})

        # Show classification results on the image
        batch_size = num_detections.shape[0]
        image = image.reshape((image.shape[1], image.shape[2], 3))
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        for batch in range(0, batch_size):
            for detection in range(0, int(num_detections[batch])):
                c = detection_classes[batch][detection]
                d = detection_boxes[batch][detection]
                s = detection_scores[batch][detection]
                draw_detection(draw, d, c)

        # Calculate the FPS
        if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
            end_time = time.time()
            fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        image = np.asarray(img)
        fps_text = "FPS = " + str(int(fps))
        text_location = (_LEFT_MARGIN, _ROW_SIZE)
        cv2.putText(
            image,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            _FONT_SIZE,
            _TEXT_COLOR,
            _FONT_THICKNESS,
        )

        cv2.imshow("VIDEO", image)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Name of image classification model.",
        required=False,
        default="models/ssd_mobilenet_v1_10.onnx",
    )
    parser.add_argument(
        "-u",
        "--rtsp-username",
        required=False,
        default="admin",
    )
    parser.add_argument(
        "-p",
        "--rtsp-password",
        help="RTSP stream password",
        required=True,
    )
    parser.add_argument(
        "--maxResults",
        help="Max of classification results.",
        required=False,
        default=3,
    )
    parser.add_argument(
        "--scoreThreshold",
        help="The score threshold of classification results.",
        required=False,
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        default=640,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        default=480,
    )
    args = parser.parse_args()

    run(
        args.model,
        args.rtsp_username,
        args.rtsp_password,
        # int(args.maxResults),
        # args.scoreThreshold,
        # int(args.numThreads),
        # bool(args.enableEdgeTPU),
        # int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
    )


if __name__ == "__main__":
    main()
