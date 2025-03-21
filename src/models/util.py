from viam.media.utils.pil import viam_to_pil_image, pil_to_viam_image
from PIL import Image
from viam.media.video import CameraMimeType
from datetime import datetime
import re

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

def merge_bounding_boxes(box1, box2, padding = 0):
    """
    Merges two bounding boxes into a single bounding box that encompasses both.

    :param box1: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
    :param box2: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max'
    :param padding: percent padding to add, in percent 0-1
    :return: A new bounding box that contains both input boxes.
    """
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

    # Uncomment the below only for testing
    #random_filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    #cropped_image.save(random_filename)
    #print(f"Image saved as: {random_filename}")

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


def sort_areas_ltr(areas, y_tolerance=0.02):
    """
    Sorts a list of areas by BoundingBox objects in Left-to-Right, Top-to-Bottom order.

    Parameters:
        areas: List of objects containing bounding box objects.
        y_tolerance (float): Allowed Y-difference to consider boxes in the same row.

    Returns:
        List of sorted BoundingBox objects.
    """
    # Step 1: Sort by (top Y, then left X)
    areas.sort(key=lambda b: (b.dims.y_min, b.dims.x_min))

    # Step 2: Group boxes into rows based on Y proximity
    rows = []
    current_row = []

    for area in areas:
        if not current_row:
            current_row.append(area)
        else:
            last_area = current_row[-1]
            if abs(area.dims.y_min - last_area.dims.y_min) <= y_tolerance:
                current_row.append(area)
            else:
                rows.append(current_row)
                current_row = [area]
    
    if current_row:
        rows.append(current_row)

    # Step 3: Sort each row Left-to-Right
    for row in rows:
        row.sort(key=lambda b: b.dims.x_min)

    # Flatten the sorted rows
    sorted_areas = [area for row in rows for area in row]
    
    return sorted_areas

def classification_to_float(classification):
    if isinstance(classification, bool):  
        return int(classification)  # Convert True -> 1, False -> 0
    elif isinstance(classification, int):  
        return float(classification)  # Convert int -> float
    return classification 


def get_group(groups, name):
    """Find a group by name."""
    return next((g for g in groups if g.name == name), None)

def avg(group):
    """Calculate the average classification value of a group's areas."""
    if not group or not group.areas:
        return 0
    values = [(a.classification if isinstance(a.classification, (int, float)) else int(a.classification)) for a in group.areas]
    return sum(values) / len(values) if values else 0

def count(group):
    """Count the sum of classification values in a group's areas."""
    if not group or not group.areas:
        return 0
    return sum(a.classification if isinstance(a.classification, (int, float)) else int(a.classification) for a in group.areas)

def avg_max(group, x):
    """Find the average of the highest classification values from the last X stored values across areas."""
    if not group or not group.areas:
        return 0
    max_values = [
        max([(v if isinstance(v, (int, float)) else int(v)) for v in area.history.get()[:x]], default=0)
        for area in group.areas
    ]
    return sum(max_values) / len(max_values) if max_values else 0

def count_max(group, x):
    """Sum the highest classification value from the last X stored values in history for each area."""
    if not group or not group.areas:
        return 0
    return sum(
        max([(v if isinstance(v, (int, float)) else int(v)) for v in area.history.get()[:x]], default=0)
        for area in group.areas
    )

def eval_area_expression(expression, groups):
    """Evaluate a logical expression based on group data."""
    expression = expression.replace("&&", " and ").replace("||", " or ")

    # Match avg(), count(), avg_max(), and count_max()
    pattern = re.compile(r"(avg|count|avg_max|count_max)\((\w+)(?:,\s*(\d+))?\)")
    matches = pattern.findall(expression)

    for func, group_name, x in matches:
        group = get_group(groups, group_name)
        if func == "avg":
            value = avg(group)
        elif func == "count":
            value = count(group)
        elif func == "avg_max":
            value = avg_max(group, int(x)) if x else 0  # Default to 0 if no X provided
        elif func == "count_max":
            value = count_max(group, int(x)) if x else 0  # Default to 0 if no X provided
        else:
            continue
        
        expression = expression.replace(f"{func}({group_name}{', ' + x if x else ''})", str(value))

    return eval(expression)