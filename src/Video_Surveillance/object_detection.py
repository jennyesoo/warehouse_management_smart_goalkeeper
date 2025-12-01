"""
Revise from darknet (https://github.com/AlexeyAB/darknet)
Original code: darknet_images.py
"""

import random
import time
import cv2
import darknet

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    # image = darknet.draw_boxes(detections, image_resized, class_colors)
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB),  detections
    return image_resized, detections

def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    width, height, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def load_network(batch_size=1, weights='./yolov4-obj_last.weights' , config_file='./cfg/yolov4-obj.cfg', data_file='./cfg/obj.data'):
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )
    return network, class_names, class_colors


def object_detection(image, network, class_names, class_colors, thresh=0.7, ext_output=False, save_labels=False):
    # random.seed(3)  # deterministic bbox colors
    # network, class_names, class_colors = darknet.load_network(
    #     config_file,
    #     data_file,
    #     weights,
    #     batch_size=batch_size
    # )

    prev_time = time.time()
    # image, detections = image_detection(
    #     image, network, class_names, class_colors, thresh
    #     )
    image_resized, detections = image_detection(
        image, network, class_names, class_colors, thresh
        )
    if save_labels:
        save_annotations(image, image, detections, class_names)
    darknet.print_detections(detections, ext_output)
    # fps = int(1/(time.time() - prev_time))
    # print("FPS: {}".format(fps))

    return image_resized, detections