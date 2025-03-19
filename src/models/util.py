from viam.media.utils.pil import viam_to_pil_image, pil_to_viam_image
from PIL import Image
from viam.media.video import CameraMimeType

def check_box_overlap(box1, box2, threshold=0.0):
    """
    Check if two bounding boxes overlap, if one (expanded) contains the other, 
    or if they are within a threshold-based expanded overlap.

    :param box1: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
    :param box2: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
    :param threshold: How close (as a fraction of width/height) two boxes can be and still count as overlapping.
    :return: True if the expanded boxes overlap or one expanded box contains the other.
    """
    # Compute box dimensions
    width1 = box1["x_max"] - box1["x_min"]
    height1 = box1["y_max"] - box1["y_min"]
    width2 = box2["x_max"] - box2["x_min"]
    height2 = box2["y_max"] - box2["y_min"]

    # Compute expansion amount based on threshold
    expand_x1 = width1 * threshold
    expand_y1 = height1 * threshold
    expand_x2 = width2 * threshold
    expand_y2 = height2 * threshold

    # Expand both boxes
    expanded_box1 = {
        "x_min": box1["x_min"] - expand_x1,
        "x_max": box1["x_max"] + expand_x1,
        "y_min": box1["y_min"] - expand_y1,
        "y_max": box1["y_max"] + expand_y1,
    }

    expanded_box2 = {
        "x_min": box2["x_min"] - expand_x2,
        "x_max": box2["x_max"] + expand_x2,
        "y_min": box2["y_min"] - expand_y2,
        "y_max": box2["y_max"] + expand_y2,
    }

    # Check if expanded_box1 contains expanded_box2
    box1_contains_box2 = (
        expanded_box1["x_min"] <= expanded_box2["x_min"] and expanded_box1["x_max"] >= expanded_box2["x_max"] and
        expanded_box1["y_min"] <= expanded_box2["y_min"] and expanded_box1["y_max"] >= expanded_box2["y_max"]
    )

    # Check if expanded_box2 contains expanded_box1
    box2_contains_box1 = (
        expanded_box2["x_min"] <= expanded_box1["x_min"] and expanded_box2["x_max"] >= expanded_box1["x_max"] and
        expanded_box2["y_min"] <= expanded_box1["y_min"] and expanded_box2["y_max"] >= expanded_box1["y_max"]
    )

    # Check for overlap using expanded boxes
    expanded_overlap = (
        expanded_box1["x_min"] < expanded_box2["x_max"] and expanded_box1["x_max"] > expanded_box2["x_min"] and
        expanded_box1["y_min"] < expanded_box2["y_max"] and expanded_box1["y_max"] > expanded_box2["y_min"]
    )

    return expanded_overlap or box1_contains_box2 or box2_contains_box1

def merge_bounding_boxes(box1, box2, padding_percent = 0):
    """
    Merges two bounding boxes into a single bounding box that encompasses both.

    :param box1: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
    :param box2: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
    :param padding_percent: padding to add, in percent 0-1
    :return: A new bounding box that contains both input boxes.
    """
    padding = 1 + padding_percent
    merged_box = {
        "x_min": min(box1.x_min, box2.x_min) - (padding * min(box1.x_min, box2.x_min)),
        "x_max": max(box1.x_max, box2.x_max) + (padding * max(box1.x_max, box2.x_max)),
        "y_min": min(box1.y_min, box2.y_min) - (padding * min(box1.y_min, box2.y_min)),
        "y_max": max(box1.y_max, box2.y_max) + (padding * max(box1.y_max, box2.y_max)),
    }
    return merged_box

def crop_viam_image(viam_image, bbox):
    image = viam_to_pil_image(viam_image)
    abs_dims = get_absolute_dims(image, bbox)

    # Crop the image (left, upper, right, lower)
    cropped_image = image.crop((abs_dims["x_min"], abs_dims["y_min"], abs_dims["x_max"], abs_dims["y_max"]))

    return pil_to_viam_image(cropped_image, CameraMimeType.JPEG)

def get_absolute_dims(image, bbox):
    width, height = image.size  # Get original image size

    # Convert relative coordinates to absolute pixel values
    x_min = int(bbox["x_min"] * width)
    x_max = int(bbox["x_max"] * width)
    y_min = int(bbox["y_min"] * height)
    y_max = int(bbox["y_max"] * height)

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }