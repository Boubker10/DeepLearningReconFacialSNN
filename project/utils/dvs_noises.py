from dataclasses import dataclass
from typing import Tuple
import numpy as np

from project.utils.drop_event import (
    drop_by_area_numpy,
    drop_by_time_numpy,
    drop_event_numpy,
)


@dataclass(frozen=True)
class EventDrop:
    """Applies EventDrop transformation from the paper "EventDrop: Data Augmentation for Event-based Learning".
        Applies one of the 4 drops of event strategies between:
            1. Identity (do nothing)
            2. Drop events by time
            3. Drop events by area
            4. Drop events randomly

        For each strategy, the ratio of dropped events are determined in the paper.

    Args:
        sensor_size (Tuple): size of the sensor that was used [W,H,P]

    Example:
        >>> transform = tonic.transforms.EventDrop(sensor_size=(128,128,2))
    """

    sensor_size: Tuple[int, int, int]

    def __call__(self, events):
        choice = np.random.randint(0, 4)
        if choice == 0:
            return events
        if choice == 1:
            duration_ratio = np.random.randint(1, 10) / 10.0
            return drop_by_time_numpy(events, duration_ratio)
        if choice == 2:
            area_ratio = np.random.randint(1, 6) / 20.0
            return drop_by_area_numpy(events, self.sensor_size, area_ratio)
        if choice == 3:
            ratio = np.random.randint(1, 10) / 10.0
            return drop_event_numpy(events, ratio)
