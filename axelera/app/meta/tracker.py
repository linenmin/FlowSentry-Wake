# Copyright Axelera AI, 2023
# Metadata for tracker
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

import numpy as np

from axelera.app.meta.classification import ClassificationMeta
from axelera.app.meta.keypoint import CocoBodyKeypointsMeta

from .. import display, eval_interfaces, plot_utils
from .base import AxTaskMeta, MetaObject, class_as_label


class TrackedObject(MetaObject):
    def __init__(self, meta, index, track_id):
        super().__init__(meta, index)
        self._track_id = track_id

    @property
    def track_id(self):
        return self._track_id

    @property
    def history(self):
        return self._meta.tracking_history[self.track_id]

    @property
    def class_id(self):
        return self._meta.class_ids[self._index]


_red = (255, 0, 0, 255)
_yellow = (255, 255, 0, 255)


def _track_id_as_label(track_id: int, labels: Optional[list] = None) -> str:
    if not labels:
        return f'id:{track_id}'
    return f"{track_id}"


# the dataclasses for each computer vision task
@dataclass(frozen=True)
class TrackerMeta(AxTaskMeta):
    """Metadata for tracker task"""

    Object: ClassVar[MetaObject] = TrackedObject

    # key is the track id, value is the bbox history
    tracking_history: Dict[int, np.ndarray] = field(default_factory=dict)
    class_ids: List[int] = field(default_factory=list)
    object_meta: Dict[str, Dict[int, AxTaskMeta]] = field(default_factory=dict)
    frame_object_meta: Dict[str, Dict[int, AxTaskMeta]] = field(default_factory=dict)
    labels: Optional[list] = field(default_factory=lambda: None, repr=False)
    labels_dict: Dict[str, list] = field(default_factory=dict)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.tracking_history) == len(
            self.class_ids
        ), f"Number of track ids {len(self.tracking_history)} does not match number of class ids {len(self.class_ids)}"

    def draw(self, draw: display.Draw):
        for idx, (track_id, bboxes) in enumerate(self.tracking_history.items()):
            cmap = draw.options.bbox_class_colors
            color = cmap.get(track_id, plot_utils.get_color(track_id))

            tag = ''
            if self.task_render_config.show_labels:
                label = ''
                if (class_id := self.class_ids[idx]) != -1:
                    label = f'{class_as_label(self.labels, class_id)}'
                track_id_label = _track_id_as_label(track_id, self.labels)
                fmt = draw.options.tracker_label_format
                tag = fmt.format(
                    label=label,
                    track_id=track_id_label,
                )
                for submeta_key, values in self.object_meta.items():
                    value = values.get(track_id)
                    if value is None:
                        continue
                    if isinstance(value, ClassificationMeta):
                        class_id = value._class_ids[0][0]
                        score = value._scores[0][0]
                        submeta_fmt = draw.options.bbox_label_format

                        sublabel = submeta_fmt.format(
                            label=class_as_label(self.labels_dict[submeta_key], class_id),
                            score=score,
                            scorep=score * 100,
                            scoreunit='%',
                        )
                        if sublabel:
                            if tag:
                                tag += f'\n{submeta_key}: {sublabel}'
                            else:
                                tag = f'{submeta_key}: {sublabel}'

            bbox = bboxes[-1]
            if np.all(bbox == 0):
                continue
            if self.task_render_config.show_annotations:
                draw.labelled_box((bbox[0], bbox[1]), (bbox[2], bbox[3]), tag, color)
                draw.trajectory(bboxes, color)
            elif tag:
                draw.labelled_box((bbox[0], bbox[1]), (bbox[0], bbox[1]), tag, color)

        for submeta_key, values in self.frame_object_meta.items():
            for track_id, value in values.items():
                if value is None:
                    continue
                if isinstance(value, CocoBodyKeypointsMeta):
                    value.draw(draw)

    @property
    def objects(self) -> List[TrackedObject]:
        if not self._objects:
            self._objects.extend(
                self.Object(self, idx, track_id)
                for idx, track_id in enumerate(self.tracking_history.keys())
            )
        return self._objects

    def to_evaluation(self):
        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")

        bboxes = []
        track_ids = []

        for idx, (track_id, bbox_history) in enumerate(self.tracking_history.items()):
            last_bbox = bbox_history[-1]

            bboxes.append(last_bbox)
            track_ids.append(track_id)

        if len(self.class_ids) > 0:
            prediction = eval_interfaces.TrackerEvalSample(
                np.array(bboxes, dtype=np.float32),
                np.array(track_ids, dtype=np.int32),
                np.array(self.class_ids, dtype=np.int32),
            )
        else:
            prediction = eval_interfaces.TrackerEvalSample()

        return prediction
