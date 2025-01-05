from collections import defaultdict
import logging
import time

import mpf_component_api as mpf
import mpf_component_util as mpf_util
from ultralytics import YOLO


logger = logging.getLogger('PuzzlePieceDetection')

def load_model():
    return YOLO("/models/puzzle.pt")

class PuzzlePieceDetection(mpf_util.VideoCaptureMixin, mpf_util.ImageReaderMixin):

    def __init__(self):
        self.model_loaded = False
        self.init_model()

    def init_model(self):
        self.model_loaded = False
        self.model = load_model()
        self.model_loaded = True

    def wait_for_model_to_load(self):
        while not self.model_loaded:
            logger.info('Waiting for model to load...')
            time.sleep(1)
        logger.info('Model loaded')

    @staticmethod
    def yolo_result_to_mpf_image_location(result):
        class_name = result.names[int(result.boxes.cls)]
        confidence = result.boxes.conf[0]
        x1, y1, x2, y2 = result.boxes.xyxy[0]
        width = x2 - x1
        height = y2 - y1
        return mpf.ImageLocation(
            x1, y1, width, height, confidence, {
                "CLASSIFICATION": class_name,
                "CLASSIFICATION CONFIDENCE LIST": f"{confidence}",
                "CLASSIFICATION LIST": f"{class_name}"
            }
        )

    def construct_classes_argument(self, job_properties):
        if job_properties["INCLUDE_EDGE"] not in ["true", "false"]:
            raise ValueError("INCLUDE_EDGE must be either 'true' or 'false'")
        if job_properties["INCLUDE_REGULAR"] not in ["true", "false"]:
            raise ValueError("INCLUDE_REGULAR must be either 'true' or 'false'")
        include_edge = job_properties["INCLUDE_EDGE"] == "true"
        include_regular = job_properties["INCLUDE_REGULAR"] == "true"
        classes = []
        class_id_by_name = {name: i for i, name in self.model.names.items()}
        if include_edge:
            classes.append(class_id_by_name["edge"])
        if include_regular:
            classes.append(class_id_by_name["regular"])
        return classes

    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)
        classes = self.construct_classes_argument(video_job.job_properties)
        self.init_model()
        tracked_objects = defaultdict(list)
        for frame_index, frame in enumerate(video_capture):
            results = self.model.track(
                frame,
                persist=True,
                conf=float(video_job.job_properties["CONFIDENCE"]),
                classes=classes,
            )[0]
            for result in results:
                if not result.boxes.is_track:
                    logger.info(f'Skipping non-track result in frame {frame_index}')
                    continue  # Sometimes the first detection of a track is not assigned a track id. If this happens, ignore it
                image_location = PuzzlePieceDetection.yolo_result_to_mpf_image_location(result)
                # print(image_location)
                object_id = int(result.boxes.id)
                tracked_objects[object_id].append({
                    "location": image_location,
                    "frame": frame_index,
                })
        tracks = []
        for locations in tracked_objects.values():
            class_name = locations[0]["location"].detection_properties["CLASSIFICATION"]
            start_frame = locations[0]["frame"]
            stop_frame = locations[-1]["frame"]
            max_confidence = max([location["location"].confidence for location in locations])
            tracks.append(
                mpf.VideoTrack(
                    start_frame,
                    stop_frame,
                    max_confidence,
                    frame_locations={
                        location["frame"]: location["location"] for location in locations
                    },
                    detection_properties={
                        "CLASSIFICATION": class_name,
                    }
                )
            )
        return tracks

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info('[%s] Received image job: %s', image_job.job_name, image_job)
        classes = self.construct_classes_argument(image_job.job_properties)
        self.init_model()
        self.wait_for_model_to_load()
        results = self.model(
            image_reader.get_image(),
            conf=float(image_job.job_properties["CONFIDENCE"]),
            classes=classes,
        )[0]
        detections = []
        for result in results:
            detections.append(PuzzlePieceDetection.yolo_result_to_mpf_image_location(result))
        return detections
