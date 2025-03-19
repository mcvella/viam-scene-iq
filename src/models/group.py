from .area import AreaClassifier, AreaClassifierBool, AreaDetectorBool, AreaDetectorCount, AreaGaze, AreaSensor

from viam.components.sensor import Sensor
from viam.services.vision import VisionClient

class Group():
    name: str
    type: str
    resource = str
    actual_resource: VisionClient|Sensor
    reference_image: str
    from_label: str
    to_label: str
    ml_class: str
    confidence: float
    areas: list[AreaClassifier|AreaClassifierBool|AreaDetectorBool|AreaDetectorCount|AreaGaze|AreaSensor]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.__dict__['areas'] = []