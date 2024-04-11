from dataclasses import dataclass
import torch

from torchvision import transforms
from torchvision.transforms import functional
import random
from project.utils.dvs_noises import EventDrop
from project.utils.transform_dvs import (
    BackgroundActivityNoise,
    ConcatTimeChannels,
    RandomFlipLR,
    RandomFlipPolarity,
    RandomTimeReversal,
    get_frame_representation,
)


class DVSTransform:
    def __init__(
        self,
        sensor_size=None,
        timesteps: int = 10,
        transforms_list=[],
        concat_time_channels=True,
        dataset=None,
    ):
        trans = []

        representation = get_frame_representation(
            sensor_size, timesteps
        )

        # BEFORE TENSOR TRANSFORMATION
        if "flip" in transforms_list:
            trans.append(RandomFlipLR(sensor_size=sensor_size))

        if "background_activity" in transforms_list:
            trans.append(
                transforms.RandomApply(
                    [BackgroundActivityNoise(severity=5, sensor_size=sensor_size)],
                    p=0.5,
                )
            )

        if "reverse" in transforms_list:
            trans.append(RandomTimeReversal(p=0.2))  # only for transformation A (not B)

        if "flip_polarity" in transforms_list:
            trans.append(RandomFlipPolarity(p=0.2))

        if "event_drop" in transforms_list:
            trans.append(EventDrop(sensor_size=sensor_size))

        # TENSOR TRANSFORMATION
        trans.append(representation)

        # if 'crop' in transforms_list:
        if "crop" in transforms_list:
            trans.append(
                transforms.RandomResizedCrop(
                    (128, 128), interpolation=0
                )
            )
        else:
            trans.append(
                transforms.Resize(
                    (128, 128), interpolation=0
                )
            )

        if "mirror" in transforms_list:
            trans.append(transforms.RandomApply([FaceMirror()], p=0.5))

        # finish by concatenating polarity and timesteps
        if concat_time_channels:
            trans.append(ConcatTimeChannels())

            self.transform = transforms.Compose(
                [
                    representation,
                    transforms.Resize(
                        (128, 128), interpolation=0
                    ),
                    ConcatTimeChannels(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    representation,
                    transforms.Resize(
                        (128, 128), interpolation=0
                    ),
                ]
            )

        self.transform = transforms.Compose(trans)

    def __call__(self, X):
        X = self.transform(X)
        return X


@dataclass(frozen=True)
class FaceMirror:
    def __call__(self, frames: torch.Tensor):  # shape (T, C, H, W)
        W = frames.shape[-1]
        if random.random() >= 0.5:  # left
            left = frames[:, :, :, 0 : W // 2]
            new_right = functional.hflip(left)
            frames[:, :, :, W // 2 :] = new_right
        else:  # right
            right = frames[:, :, :, W // 2 :]
            new_left = functional.hflip(right)
            frames[:, :, :, 0 : W // 2] = new_left

        return frames
