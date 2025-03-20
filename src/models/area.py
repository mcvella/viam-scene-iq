from viam.components.sensor import Sensor
from viam.services.vision import VisionClient
from .util import *
from viam.media.utils.pil import viam_to_pil_image

import re

class AreaDims():
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
    
    def to_dict(self):
        """Convert AreaDims instance to a dictionary."""
        return vars(self)  
    
class AreaGaze():
    type: str="gaze"
    index: int
    dims: AreaDims
    to_dims: AreaDims
    full_dims: AreaDims
    classification: bool
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.dims = AreaDims()
        self.to_dims = AreaDims()
        self.full_dims = AreaDims()

    def to_dict(self):
        """Convert the object to a dictionary, ensuring AreaDims instances are converted."""
        return {
            key: value.to_dict() if isinstance(value, AreaDims) else value
            for key, value in self.__dict__.items()
        }
    
    async def get_classification(self, logger, resource: VisionClient, image):
        detections = await resource.get_detections(crop_viam_image(image, vars(self.full_dims)))
        matches = {}
        pil_image = viam_to_pil_image(image)
        full_abs_dims = get_absolute_dims(pil_image, vars(self.full_dims))
        for d in detections:
            match = re.fullmatch(r"(face|gaze)_(.*)", d.class_name)
            if match:
                match_type = match.group(1)
                match_label = "match_" + match.group(2)
                if not match_label in matches:
                    matches[match_label] = {}
                matches[match_label][match_type] = {"x_min": full_abs_dims["x_min"] + d.x_min, "x_max": full_abs_dims["x_min"] + d.x_max, "y_min": full_abs_dims["y_min"] + d.y_min, "y_max": full_abs_dims["y_min"] + d.y_max}
        
        for match in matches:
            if "face" in matches[match] and "gaze" in matches[match]:
                # detection dimensions are returned in absolute while we have relative from the reference image
                if check_box_overlap(matches[match]["face"], get_absolute_dims(pil_image, vars(self.dims)), .25) and check_box_overlap(matches[match]["gaze"], get_absolute_dims(pil_image, vars(self.to_dims)), .5):
                    self.classification = True
                    return True
        self.classification = False
        return False

class AreaDetectorBool():
    type: str="detector_bool"
    index: int
    dims: AreaDims
    classification: bool

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.dims = AreaDims()

    async def get_classification(self, logger, resource, image, ml_class, confidence):
        detections = await resource.get_detections(crop_viam_image(image, vars(self.dims)))
        for d in detections:
            if (d.class_name == ml_class) and (d.confidence >= confidence):
                self.classification = True
                return True
        self.classification = False
        return False

class AreaDetectorCount():
    type: str="detector_count"
    index: int
    dims: AreaDims
    classification: int

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.dims = AreaDims()

    async def get_classification(self, logger, resource, image, ml_class, confidence):
        detections = await resource.get_detections(crop_viam_image(image, vars(self.dims)))
        count = 0
        for d in detections:
            if (d.class_name == ml_class) and (d.confidence >= confidence):
                count = count + 1
        self.classification = count
        return count

class AreaClassifier():
    type: str="classifier"
    index: int
    dims: AreaDims
    classification: str

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.dims = AreaDims()
  
    async def get_classification(self, logger, resource, image):
        classifications = await resource.get_classifications(crop_viam_image(image, vars(self.dims)), 1)
        classification = ""
        if len(classifications) > 0:
           classification = classifications[0].class_name
        self.classification = classification
        return classification
    
class AreaClassifierBool():
    type: str="classifier_bool"
    index: int
    dims: AreaDims
    classification: bool

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.dims = AreaDims()

    async def get_classification(self, logger, resource, image, ml_class, confidence):
        # we only look at the top 5 classifications, which could be artificially limiting?
        classifications = await resource.get_classifications(crop_viam_image(image, vars(self.dims)), 5)
        if len(classifications) > 0:
           for c in classifications:
               if (c.class_name == ml_class) and (c.confidence >= confidence):
                    self.classification = True
                    return True
        self.classification = False
        return False
    
class AreaSensor():
    type: str="sensor"
    index: int
    dims: AreaDims
    classification: any

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.dims = AreaDims()

    async def get_classification(self, logger, resource):
        reading = resource.get_readings()
        self.classification = reading
        return reading