# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import numpy as np
import torch
import warnings

from typing import Any, Dict, List, Tuple, Union

from cubifyanything.measurement import BaseMeasurementInfo
from cubifyanything.orientation import ImageOrientation, get_orientation, rotate_pose

# Extends some of the ideas of D2's "Instances" to more broad sensors.
class SensorInfo(object):
    def __init__(self, **kwargs):
        self._measurements: Dict[str, Any] = {}
        self._other = {}
        self._meta_keys = []        

        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name in ["_other", "_measurements", "_RT", "_meta_keys"]:
            super(SensorInfo, self).__setattr__(name, val)
        elif name.startswith("_"):
            self._other[name] = val
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_other":
            return self.__getattribute__("_other")

        if name.startswith("_"):
            if name in self._other:
                return self._other[name]

            return self.__getattribute__(name)

        if name not in self._measurements:
            raise AttributeError("Cannot find field '{}' in the given measurements!".format(name))

        return self._measurements[name]

    def __getstate__(self):
        return {"_measurements": self._measurements, "_other": self._other, "_meta_keys": self._meta_keys}

    def __setstate__(self, s):
        self._measurements = s["_measurements"]
        self._other = s["_other"]
        self._meta_keys = s["_meta_keys"]

    @property
    def ts(self):
        if isinstance(self, PosedSensorInfo):
            return self._RT_meta.ts

        # TODO: Take first measurement and ask for ts?
        return None    

    def translate(self, t):
        raise NotImplementedError

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)

        if len(self._measurements):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a measurement of length {}".format(data_len, len(self))

        self._measurements[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._measurements

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._measurements[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._measurements[name]

    @classmethod
    def cat(self, sensor_list):
        # TODO: Flesh this out better.
        measurement_names = sensor_list[0].get_measurements().keys()
        measurements = {}

        for measurement_name in measurement_names:
            info_list = [getattr(sensor_list_, measurement_name) for sensor_list_ in sensor_list]
            measurements[measurement_name] = type(info_list[0]).cat(info_list)

        return type(sensor_list[0])(**measurements)

    def __len__(self) -> int:
        for v in self._measurements.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()

    def get_measurements(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        # for now, only return subclasses of MeasurementInfo
        return { k: m for k, m in self._measurements.items() if isinstance(m, (MeasurementInfo,)) }

    def orient(self, current_orientation, target_orientation):
        new_sensor_info = type(self)()
        new_sensor_info._other = dict(self._other)

        # Save this for the ability to restore?
        new_sensor_info._original_orientation = current_orientation

        # One of these needs to be UPRIGHT for now.
        if (current_orientation != ImageOrientation.UPRIGHT) and (target_orientation != ImageOrientation.UPRIGHT):
            raise NotImplementedError

        for measurement_name, measurement in self._measurements.items():
            # TODO: fix this as an _other_?
            if measurement_name == "RT":
                new_sensor_info.RT = rotate_pose(self.RT, current_orientation, target=target_orientation)
            elif measurement_name == "ts":
                new_sensor_info.ts = self.ts.clone()
            elif isinstance(measurement, BaseMeasurementInfo):
                setattr(new_sensor_info, measurement_name, measurement.orient(current_orientation, target_orientation))

        if isinstance(self, PosedSensorInfo):
            new_sensor_info._RT = self._RT.clone()

        # Make sure we continue to use the override.
        if hasattr(self, "_orientation"):
            setattr(new_sensor_info, "_orientation", target_orientation)

        return new_sensor_info

    def to(self, *args: Any, **kwargs: Any) -> "SensorInfo":        
        ret = type(self)()
        for k, v in self._measurements.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)

        ret._other = dict(self._other)
        for meta_key in self._meta_keys:
            ret._meta_keys.append(meta_key)
            setattr(ret, meta_key, getattr(self, meta_key))
        
        return ret

# TODO: this should enforce "RT" (i.e. pose) existing.
class PosedSensorInfo(SensorInfo):
    @property
    def orientation(self):
        # Allow override.
        if hasattr(self, "_orientation"):
            return self._orientation

        # for now, assume we're dealing with a single orientation. majority vote.
        if len(self.RT) == 1:
            return ImageOrientation(get_orientation(self.RT)[-1].item())

        orientations = get_orientation(self.RT).cpu().numpy()
        unique_orientations, counts = np.unique(orientations, return_counts=True)
        most_frequent_orientation = unique_orientations[np.argmax(counts)]

        return ImageOrientation(most_frequent_orientation)

    @property
    def device(self):
        return self.RT.device

    def set(self, name: str, value: Any) -> None:
        if name == "RT":
            # only write if we don't already have 
            if not hasattr(self, "_RT"):
                self._RT = value.clone()

        super(PosedSensorInfo, self).set(name, value)

    def apply_transform(self, transform_4x4):
        new_sensor_info = PosedSensorInfo()
        new_sensor_info._RT = self._RT.clone()
        new_sensor_info._other = dict(self._other)

        for measurement_name, measurement in self._measurements.items():
            if measurement_name == "RT":
                new_sensor_info.RT = transform_4x4 @ self.RT
            elif measurement_name == "ts":
                new_sensor_info.ts = self.ts.clone()
            elif hasattr(measurement, "apply_transform"):
                setattr(new_sensor_info, measurement_name, measurement.apply_transform(transform_4x4))
            else:

                setattr(new_sensor_info, measurement_name, measurement)

        return new_sensor_info

    def translate(self, t):
        translation_4x4 = torch.eye(4)[None, ...].to(t.device)
        translation_4x4[:, :3, -1] = t

        return self.apply_transform(translation_4x4)

    @classmethod
    def cat(cls, sensor_list):
        new_sensor_info = SensorInfo.cat(sensor_list)
        new_sensor_info.RT = torch.cat([sensor_info.RT for sensor_info in sensor_list])

        return new_sensor_info

class SensorArrayInfo(object):
    def __init__(self, **kwargs: Any):
        self._sensors: Dict[str, SensorInfo] = {}
        self._rel_transforms: Dict[Tuple[str, str], torch.Tensor] = {}

        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_sensors" or name not in self._sensors:
            raise AttributeError("Cannot find field '{}' in the given sensors!".format(name))

        return self._sensors[name]

    def __getstate__(self):
        return self._sensors

    def __setstate__(self, d):
        self._sensors = d

    def set(self, name: str, value: Any) -> None:
        self._sensors[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._sensors

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._sensors[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._sensors[name]

    # This is not really always a good idea because sensor's don't _have_ to
    # have the same length (although they often do).
    def uniform_length(self) -> int:
        for v in self._sensors.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()

    def to(self, *args: Any, **kwargs: Any) -> "SensorArrayInfo":
        ret = type(self)()
        for k, v in self._sensors.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)

        return ret
    
