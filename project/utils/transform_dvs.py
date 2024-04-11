from dataclasses import dataclass
from typing import Tuple
from einops import rearrange
import torch

from torchvision import transforms
from tonic.transforms import functional
import numpy as np


@dataclass(frozen=True)
class RandomTimeReversal:
    """Temporal flip is defined as:

        .. math::
           t_i' = max(t) - t_i

           p_i' = -1 * p_i

    Parameters:
        p (float): probability of performing the flip
    """

    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "t" and "p" in events.dtype.names
        if np.random.rand() < self.p:
            events["t"] = np.max(events["t"]) - events["t"]
            # events = events[np.argsort(events["t"])]
            events = np.flip(events)
            # events["p"] *= -1
            events["p"] = np.logical_not(events["p"])  # apply to boolean (inverse)
        return events


@dataclass(frozen=True)
class RandomFlipPolarity:
    """Flips polarity of individual events with p.
    Changes polarities 1 to -1 and polarities [-1, 0] to 1

    Parameters:
        p (float): probability of flipping individual event polarities
    """

    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "p" in events.dtype.names
        # flips = np.ones(len(events))
        probs = np.random.rand(len(events))
        mask = probs < self.p
        events["p"][mask] = np.logical_not(events["p"][mask])
        return events


@dataclass(frozen=True)
class RandomFlipLR:
    """Flips events in x. Pixels map as:

        x' = width - x

    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        p (float): probability of performing the flip
    """

    sensor_size: Tuple[int, int, int] = None
    p: float = 0.5

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size

        events = events.copy()
        assert "x" in events.dtype.names
        if np.random.rand() <= self.p:
            events["x"] = sensor_size[0] - 1 - events["x"]
        return events
    

def get_sensor_size(events: np.ndarray):
    return events["x"].max() + 1, events["y"].max() + 1, 2  # H,W,2


@dataclass(frozen=True)
class BackgroundActivityNoise:
    severity: int
    sensor_size: Tuple[int, int, int] = None

    def __call__(self, events):
        c = [0.005, 0.01, 0.03, 0.10, 0.2][
            self.severity - 1
        ]  # percentage of events to add in noise
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size
        n_noise_events = int(c * len(events))
        noise_events = np.zeros(n_noise_events, dtype=events.dtype)
        for channel in events.dtype.names:
            event_channel = events[channel]
            if channel == "x":
                low, high = 0, sensor_size[0]
            if channel == "y":
                low, high = 0, sensor_size[1]
            if channel == "p":
                low, high = 0, sensor_size[2]
            if channel == "t":
                low, high = events["t"].min(), events["t"].max()

            if channel == "p":
                noise_events[channel] = np.random.choice(
                    [True, False], size=n_noise_events
                )
            else:
                noise_events[channel] = np.random.uniform(
                    low=low, high=high, size=n_noise_events
                )
        events = np.concatenate((events, noise_events))
        new_events = events[np.argsort(events["t"])]
        # new_events['p'] = events['p']

        return new_events


def get_frame_representation(sensor_size, timesteps):
    return transforms.Compose(
        [
            CustomToFrame(timesteps=timesteps, sensor_size=sensor_size),
            BinarizeFrame(),
        ]
    )


@dataclass(frozen=True)
class CustomToFrame:
    timesteps: int
    sensor_size: Tuple[int, int, int] = None
    event_count: int = 2500

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size

        if len(events) // self.event_count >= self.timesteps:
            x = functional.to_frame_numpy(
                events=events,
                sensor_size=sensor_size,
                time_window=None,
                event_count=self.event_count,
                n_time_bins=None,
                n_event_bins=None,
                overlap=0.0,
                include_incomplete=False,
            )

            if x.shape[0] > self.timesteps:
                gap = int((x.shape[0] - self.timesteps) / 2)
                return x[gap : gap + self.timesteps]
            else:
                return x
        else:
            return functional.to_frame_numpy(
                events=events,
                sensor_size=sensor_size,
                time_window=None,
                event_count=None,
                n_time_bins=None,
                n_event_bins=self.timesteps,
                overlap=0.0,
                include_incomplete=False,
            )


@dataclass(frozen=True)
class BinarizeFrame:
    def __call__(self, x):
        x = (x > 0).astype(np.float32)
        x = torch.from_numpy(x)
        return x


@dataclass(frozen=True)
class ConcatTimeChannels:
    def __call__(self, x):
        x = rearrange(
            x, "frames polarity height width -> (frames polarity) height width"
        )
        return x
    