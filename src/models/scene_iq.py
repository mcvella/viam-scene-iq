from typing import (Any, ClassVar, Dict, Final, List, Mapping, Optional,
                    Sequence, cast)

from typing_extensions import Self
from viam.components.sensor import *
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, ValueTypes, struct_to_dict
from viam.services.vision import VisionClient
from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions
from viam.proto.app.data import BinaryID
from viam.components.camera import Camera

from .group import Group
from .area import *
from .util import *

import os
import asyncio

GROUP_GLOBAL = {}

class SceneIq(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("mcvella", "scene-iq"), "scene-iq")
    group_states: list[Group] = []
    camera: Camera
    area_dims_calculated: bool = False
    max_readings_gap_secs: float = 1

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        return super().new(config, dependencies)

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        deps = []

        attributes = struct_to_dict(config.attributes)

        groups = attributes.get("groups", [])
        for group in groups:
            if "resource" in group:
                deps.append(group["resource"])
            else:
                raise Exception(f"A resource name for group {group["name"]} must be defined")
            
        if (len(groups) == 0):
            raise Exception(f"At least one group must be configured in 'groups'")
        
        camera = attributes.get("camera", "")
        if camera != "":
            deps.append(camera)
        else:
            raise Exception(f"A camera resource name must be defined")
        
        return deps

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        
        # reset this to force area dimensions to be reset on first call
        self.area_dims_calculated = False
        self.group_states: list[Group] = []
        
        attributes = struct_to_dict(config.attributes)

        self.max_readings_gap_secs = attributes.get("max_readings_gap_secs", 1)

        # set up each group, instantiating the correct resource client
        g: Group
        for group in attributes.get("groups", []):
            g = Group(**group)
            if g.type in ["gaze", "detector_bool", "detector_count", "classifier", "classifier_bool"]:
                resource_dep = dependencies[VisionClient.get_resource_name(g.resource)]
                g.actual_resource = cast(VisionClient, resource_dep)
            else:
                resource_dep = dependencies[Sensor.get_resource_name(g.resource)]
                g.actual_resource = cast(Sensor, resource_dep)              
            self.group_states.append(g)
            
        camera_name = attributes.get("camera", "")
        camera_dep = dependencies[Camera.get_resource_name(camera_name)]
        self.camera = cast(Camera, camera_dep)

        # allow access at the global level by name so a vision service can also be exposed
        GROUP_GLOBAL[config.name] = self.group_states
        
        return super().reconfigure(config, dependencies)


    async def viam_connect(self) -> ViamClient:
        dial_options = DialOptions.with_api_key( 
            api_key=os.getenv('VIAM_API_KEY'),
            api_key_id=os.getenv('VIAM_API_KEY_ID')
        )

        return await ViamClient.create_from_dial_options(dial_options)
    
    async def calculate_area_dims(self):
        self.app_client = await self.viam_connect()

        for group in self.group_states:
            group.areas = []
            image_binary_id = BinaryID(
                file_id=group.reference_image,
                organization_id=os.getenv('VIAM_PRIMARY_ORG_ID'),
                location_id=os.getenv('VIAM_LOCATION_ID')
            )

            binary_data = await self.app_client.data_client.binary_data_by_ids(binary_ids=[image_binary_id])

            # we want to sort ltr so we store them first
            areas = []
            # store any "to" dimensions for gaze detection so we can match them to the "from" afterwards
            to_dims = []

            for bbox in binary_data[0].metadata.annotations.bboxes:
                if bbox.label == group.from_label:
                    match group.type:
                        case "gaze":
                            area = AreaGaze()
                        case "detector_bool":
                            area = AreaDetectorBool()
                        case "detector_count":
                            area = AreaDetectorCount()
                        case "classifier":
                            area = AreaClassifier()
                        case "classifier_bool":
                            area = AreaClassifierBool()
                        case "sensor":
                            area = AreaSensor()
                    
                    area.dims.x_min = bbox.x_min_normalized
                    area.dims.x_max = bbox.x_max_normalized
                    area.dims.y_min = bbox.y_min_normalized
                    area.dims.y_max = bbox.y_max_normalized
                    areas.append(area)
                elif (group.to_label != "") and (group.type == "gaze") and (bbox.label == group.to_label):
                    dims = {
                        "x_min": bbox.x_min_normalized,
                        "x_max": bbox.x_max_normalized,
                        "y_min": bbox.y_min_normalized,
                        "y_max": bbox.y_max_normalized
                    }
                    to_dims.append(dims)

            group.areas = sort_areas_ltr(areas, 0.07)

            # match "from" and "to" areas for gaze
            if group.type == "gaze":
                for f in group.areas:
                    for t in to_dims:
                        if check_box_overlap(vars(f.dims), t):
                            f.to_dims = AreaDims(**t)
                            f.full_dims = AreaDims(**merge_bounding_boxes(f.dims, f.to_dims, 0.03))
                            break

        self.area_dims_calculated = True

    async def get_readings(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, SensorReading]:
        
        if not self.area_dims_calculated:
            await self.calculate_area_dims()

        image = await self.camera.get_image()

        tasks = []
        for g in self.group_states:
            i = 0
            for a in g.areas:
                # add an ordering index, this should stay static
                a.index = i
                i = i + 1
                match a.type:
                    case "gaze":
                        tasks.append(asyncio.create_task(a.get_classification(self.logger, g.actual_resource, image)))
                    case "detector_bool":
                        tasks.append(asyncio.create_task(a.get_classification(self.logger, g.actual_resource, image, g.ml_class, g.confidence)))
                    case "detector_count":
                        tasks.append(asyncio.create_task(a.get_classification(self.logger, g.actual_resource, image, g.ml_class, g.confidence)))
                    case "classifier":
                        tasks.append(asyncio.create_task(a.get_classification(self.logger, g.actual_resource, image)))
                    case "classifier_bool":
                        tasks.append(asyncio.create_task(a.get_classification(self.logger, g.actual_resource, image, g.ml_class, g.confidence)))
                    case "sensor":
                        tasks.append(asyncio.create_task(a.get_classification(self.logger, g.actual_resource)))
        await asyncio.gather(*tasks)
        to_return = {}

        for g in self.group_states:
            to_return[g.name] = {
                "name": g.name
            }
            to_return[g.name]["areas"] = []
            for a in g.areas:
                 to_return[g.name]["areas"].append({"index": a.index, "classification": a.classification})
        
        return to_return
    
    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[Geometry]:
        self.logger.error("`get_geometries` is not implemented")
        raise NotImplementedError()
