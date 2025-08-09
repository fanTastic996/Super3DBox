# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import numpy as np
import torch

from typing import Any, Dict, List, Tuple, Union

from cubifyanything.orientation import ImageOrientation, rotate_K

class BaseMeasurementInfo(object):
    def __init__(self, meta=None, **kwargs):
        super(BaseMeasurementInfo, self).__init__()
        self.meta = meta

    @property
    def ts(self):
        if (self.meta is not None) and hasattr(self.meta, "ts"):
            return self.meta.ts

        return None

class MeasurementInfo(BaseMeasurementInfo):
    pass

class ImageMeasurementInfo(MeasurementInfo):
    def __init__(self, size, K, meta=None, original_size=None):
        super(ImageMeasurementInfo, self).__init__(meta=meta)
        self.size = size
        if isinstance(self.size, torch.Tensor) and not torch.jit.is_tracing():
            self.size = (self.size[0].item(), self.size[1].item())

        self.original_size = original_size or self.size

        # check for normalized.
        if ((K[..., 2] >= 0) & (K[..., 2] < 1)).all():
            raise ValueError("Normalized intrinsics are not supported")

        # No float64 support on MPS.
        self.K = K.float()

    @property
    def device(self):
        return self.K.device

    def _get_fields(self):
        # Don't support anything fancy for now.
        return dict(
            size=torch.tensor(self.size),
            K=self.K)

    def __len__(self):
        return len(self.K)

    def __getitem__(self, item):
        ret = type(self)(self.size, self.K.__getitem__(item), meta=self.meta, original_size=self.original_size)
        return ret

    def to(self, *args: Any, **kwargs: Any) -> "ImageMeasurementInfo":
        ret = type(self)(self.size, self.K.to(*args, **kwargs), meta=self.meta, original_size=self.original_size)
        return ret

    @classmethod
    def cat(self, info_list):
        return type(info_list[0])(
            size=info_list[0].size,
            K=torch.cat([info_.K for info_ in info_list]),
        )

    def _get_oriented_size(self, current_orientation, target_orientation, size):
        if (target_orientation != ImageOrientation.UPRIGHT) and (current_orientation != ImageOrientation.UPRIGHT):
            raise NotImplementedError

        if ((current_orientation, target_orientation) in [
                (ImageOrientation.UPRIGHT, ImageOrientation.UPRIGHT),
                (ImageOrientation.UPSIDE_DOWN, ImageOrientation.UPRIGHT),
                (ImageOrientation.UPRIGHT, ImageOrientation.UPSIDE_DOWN),
                (ImageOrientation.LEFT, ImageOrientation.RIGHT),
                (ImageOrientation.RIGHT, ImageOrientation.LEFT)
        ]):
            # Nothing changes.
            new_size = size
        else:
            # Swap.
            new_size = (size[1], size[0])

        return new_size

    def orient(self, current_orientation, target_orientation):
        if (target_orientation != ImageOrientation.UPRIGHT) and (current_orientation != ImageOrientation.UPRIGHT):
            raise NotImplementedError

        new_K = rotate_K(self.K, current_orientation, self.size, target=target_orientation)
        new_size = self._get_oriented_size(current_orientation, target_orientation, self.size)

        ret = type(self)(
            new_size,
            new_K,
            meta=self.meta,
            original_size=self._get_oriented_size(current_orientation, target_orientation, self.original_size))

        return ret

    def rescale(self, factor):
        old_size = self.size
        new_size = (int(old_size[0] * factor), int(old_size[1] * factor))

        new_K = self.K.clone()
        new_K[..., :2, :] = new_K[..., :2, :] * factor

        return type(self)(new_size, new_K, meta=self.meta, original_size=self.original_size)

    def resize(self, new_size):
        if isinstance(new_size, float):
            return self.rescale(new_size)

        width_scale = new_size[0] / self.size[0]
        height_scale = new_size[1] / self.size[1]

        # Might be some some pixel errors.
        if not np.isclose(height_scale, width_scale, atol=0.025):
            print(f"Rescaling from {self.size} to {new_size}. This does not seem uniform but may be due to discretization error.")

        result = self.rescale(height_scale)
        # Even if it's not the best idea, always make sure the given size is
        # reflected.
        result.size = tuple(new_size)
        return result
        
class DepthMeasurementInfo(ImageMeasurementInfo):
    def normalize(self, parameters):
        return WhitenedDepthMeasurementInfo(
            size=self.size,
            K=self.K,
            meta=self.meta,
            parameters=parameters,
            original_size=self.original_size)

class WhitenedDepthMeasurementInfo(DepthMeasurementInfo):
    def __init__(self, size, K, meta=None, parameters=None, original_size=None):
        super(WhitenedDepthMeasurementInfo, self).__init__(size, K, meta=meta, original_size=original_size)

        # Whitening parameters.
        self.parameters = parameters

    def _get_fields(self):
        return dict(
            size=torch.tensor(self.size),
            K=self.K,
            parameters=self.parameters)
    
