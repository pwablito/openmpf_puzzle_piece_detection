import logging

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('PuzzlePieceDetection')

class PuzzlePieceDetection(mpf_util.VideoCaptureMixin, mpf_util.ImageReaderMixin):

    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)
        raise NotImplementedError

    def get_detections_from_image(self, image_job):
        logger.info('[%s] Received image job: %s', image_job.job_name, image_job)
        raise NotImplementedError
