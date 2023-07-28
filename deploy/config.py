import os

PIPELINE_NAME = "medical-image-segmentation"

PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)

METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

from absl import logging

logging