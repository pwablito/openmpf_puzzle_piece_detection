import logging

import mpf_component_api as mpf
import mpf_component_util as mpf_util
from ultralytics import YOLO


logger = logging.getLogger('PuzzlePieceDetection')

def load_model():
    return YOLO("/models/puzzle.pt")

class PuzzlePieceDetection(mpf_util.VideoCaptureMixin, mpf_util.ImageReaderMixin):

    def __init__(self):
        self.model = load_model()

    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)
        results = self.model(video_capture)
        logger.info(results)
        return []

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info('[%s] Received image job: %s', image_job.job_name, image_job)
        results = self.model(image_reader.get_image())
        logger.info(results)
        return []
