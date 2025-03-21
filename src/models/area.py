from viam.services.vision import VisionClient
from .util import *
from viam.media.utils.pil import viam_to_pil_image

import re
from collections import deque

class AreaDims:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
    
    def to_dict(self):
        return vars(self)  

class RingBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)  # Fixed-size buffer

    def append(self, item):
        """Adds an item to the buffer if it's not None, overwriting the oldest if full."""
        if item is not None:
            self.buffer.append(item)

    def get(self):
        """Returns a reversed list of elements in the buffer (newest first)."""
        return list(reversed(self.buffer))

    def __str__(self):
        """String representation of the buffer for print()."""
        return str(self.get())

    def __repr__(self):
        """Developer-friendly representation."""
        return f"RingBuffer({self.get()})"

class ClassificationMixin:
    def __init__(self, buffer_size=20):
        self.history = RingBuffer(buffer_size)
        self._classification = None

    @property
    def classification(self):
        return self._classification

    @classification.setter
    def classification(self, value):
        self._classification = value
        self.history.append(self._classification)

class AreaGaze(ClassificationMixin):
    type: str = "gaze"
    index: int
    dims: AreaDims
    to_dims: AreaDims
    full_dims: AreaDims

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.dims = AreaDims()
        self.to_dims = AreaDims()
        self.full_dims = AreaDims()

    async def get_classification(self, logger, resource: VisionClient, image):
        detections = await resource.get_detections(crop_viam_image(image, vars(self.full_dims)))
        matches = {}
        pil_image = viam_to_pil_image(image)
        full_abs_dims = get_absolute_dims(pil_image, vars(self.full_dims))
        
        for d in detections:
            match = re.fullmatch(r"(face|gaze)_(.*)", d.class_name)
            if match:
                match_type, match_label = match.groups()
                match_label = "match_" + match_label
                matches.setdefault(match_label, {})[match_type] = {
                    "x_min": full_abs_dims["x_min"] + d.x_min,
                    "x_max": full_abs_dims["x_min"] + d.x_max,
                    "y_min": full_abs_dims["y_min"] + d.y_min,
                    "y_max": full_abs_dims["y_min"] + d.y_max,
                }
        
        for match in matches.values():
            if "face" in match and "gaze" in match:
                if (check_box_overlap(match["face"], get_absolute_dims(pil_image, vars(self.dims)), 0.25) and
                        check_box_overlap(match["gaze"], get_absolute_dims(pil_image, vars(self.to_dims)), 0.5)):
                    self.classification = True
                    return True
        self.classification = False
        return False

class AreaDetectorBool(ClassificationMixin):
    type: str = "detector_bool"
    index: int
    dims: AreaDims

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.dims = AreaDims()

    async def get_classification(self, logger, resource, image, ml_class, confidence):
        detections = await resource.get_detections(crop_viam_image(image, vars(self.dims)))
        self.classification = any(d.class_name == ml_class and d.confidence >= confidence for d in detections)
        return self.classification

class AreaDetectorCount(ClassificationMixin):
    type: str = "detector_count"
    index: int
    dims: AreaDims

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.dims = AreaDims()

    async def get_classification(self, logger, resource, image, ml_class, confidence):
        detections = await resource.get_detections(crop_viam_image(image, vars(self.dims)))
        self.classification = sum(1 for d in detections if d.class_name == ml_class and d.confidence >= confidence)
        return self.classification

class AreaClassifier(ClassificationMixin):
    type: str = "classifier"
    index: int
    dims: AreaDims

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.dims = AreaDims()

    async def get_classification(self, logger, resource, image):
        classifications = await resource.get_classifications(crop_viam_image(image, vars(self.dims)), 1)
        self.classification = classifications[0].class_name if classifications else ""
        return self.classification

class AreaClassifierBool(ClassificationMixin):
    type: str = "classifier_bool"
    index: int
    dims: AreaDims

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.dims = AreaDims()

    async def get_classification(self, logger, resource, image, ml_class, confidence):
        classifications = await resource.get_classifications(crop_viam_image(image, vars(self.dims)), 5)
        self.classification = any(c.class_name == ml_class and c.confidence >= confidence for c in classifications)
        return self.classification
