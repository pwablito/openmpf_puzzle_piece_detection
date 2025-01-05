from collections import defaultdict
import logging

import mpf_component_api as mpf
import mpf_component_util as mpf_util
from ultralytics import YOLO


logger = logging.getLogger('PuzzlePieceDetection')

def load_model():
    return YOLO("/models/puzzle.pt")

class PuzzlePieceDetection(mpf_util.VideoCaptureMixin, mpf_util.ImageReaderMixin):

    def __init__(self):
        self.init_model()

    def init_model(self):
        self.model = load_model()


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

    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)
        self.init_model()  # Needed to reset track ids
        tracks = []
        return tracks

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info('[%s] Received image job: %s', image_job.job_name, image_job)
        results = self.model(
            image_reader.get_image(),
            conf=float(image_job.job_properties["CONFIDENCE"])
        )[0]
        detections = []
        for result in results:
            detections.append(PuzzlePieceDetection.yolo_result_to_mpf_image_location(result))
        return detections
