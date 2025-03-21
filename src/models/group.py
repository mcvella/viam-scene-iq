from .area import AreaClassifier, AreaClassifierBool, AreaDetectorBool, AreaDetectorCount, AreaGaze

from viam.services.vision import VisionClient

class Group():
    name: str
    type: str
    resource = str
    actual_resource: VisionClient
    reference_image: str
    from_label: str = ""
    to_label: str = ""
    ml_class: str = ""
    confidence: float = 0.7
    areas: list[AreaClassifier|AreaClassifierBool|AreaDetectorBool|AreaDetectorCount|AreaGaze]

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        self.__dict__['areas'] = []