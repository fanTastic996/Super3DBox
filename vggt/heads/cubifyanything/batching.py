# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import copy
import torch

from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from typing_extensions import TypeAlias

from cubifyanything.measurement import (
    MeasurementInfo,
    DepthMeasurementInfo,
    ImageMeasurementInfo)

from cubifyanything.sensor import (
    SensorInfo,
    PosedSensorInfo)

from cubifyanything.imagelist import ImageList
from cubifyanything.instances import Instances3D

T = TypeVar("T")
I = TypeVar("I", bound=MeasurementInfo)
S = TypeVar("S", bound=SensorInfo)

class Measurement(Generic[T, I, S]):
    def __init__(self, data: T, info: I, sensor: S):
        self.data = data
        self.info = info
        self.sensor = sensor

    # This is painful, but stems from lack of multiple dispatch.
    @classmethod
    def batch(cls, args: List["Measurement"], **kwargs) -> "BatchedMeasurement":
        if isinstance(args[0].info, (DepthMeasurementInfo,)):
            return BatchedPosedDepth(
                ImageList.from_tensors(
                    [a.data for a in args],
                    **kwargs),
                [a.info for a in args],
                [a.sensor for a in args])
        elif isinstance(args[0].info, (ImageMeasurementInfo,)):
            return BatchedPosedImage(
                ImageList.from_tensors(
                    [a.data for a in args],
                    **kwargs),
                [a.info for a in args],
                [a.sensor for a in args])
        else:
            raise NotImplementedError

    def to(self, *args: Any, **kwargs: Any) -> "Measurement":
        return self.__orig_class__(
            self.data.to(*args, **kwargs),
            self.info.to(*args, **kwargs),
            self.sensor.to(*args, **kwargs))

class BatchedMeasurement(Generic[T, I, S]):
    def __init__(self, data: T, info: List[I], sensor: List[S]):
        self.data = data
        self.info = info
        self.sensor = sensor

    @property
    def padding(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        # TODO: Also give data back (sliced).
        return self.__orig_class__(
            data=self.data if isinstance(self.data, ImageList) else self.data[index],
            info=self.info[index],
            sensor=self.sensor[index])

    # For now, only shallow copy sensor itself (since has recursive references).
    def clone(self):
        return self.__orig_class__(
            [data_.clone() if hasattr(data_, "clone") else copy.copy(data_) for data_ in self.data],
            [info_.clone() for info_ in self.info],
            copy.copy(self.sensor))

PosedImage: TypeAlias = Measurement[torch.Tensor, ImageMeasurementInfo, PosedSensorInfo]
PosedDepth: TypeAlias = Measurement[torch.Tensor, DepthMeasurementInfo, PosedSensorInfo]

BatchedPosedImage: TypeAlias = BatchedMeasurement[ImageList, ImageMeasurementInfo, PosedSensorInfo]
BatchedPosedDepth: TypeAlias = BatchedMeasurement[ImageList, DepthMeasurementInfo, PosedSensorInfo]

Sensors: TypeAlias = Dict[str, Dict[str, Measurement]]
BatchedSensors: TypeAlias = Dict[str, Dict[str, BatchedMeasurement]]
BatchedPosedSensor: TypeAlias = Dict[str, Union[BatchedPosedImage, BatchedPosedDepth]]
