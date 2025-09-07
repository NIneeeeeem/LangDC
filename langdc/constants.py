import os
from distutils.util import strtobool

# Configuration Constants
CHUNK_SIZE = 4  # Video chunk size for InternVideo2-Stage2_1B-224p-f4 which is trained using 4 frames per video
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", 16))  # Number of video frames (if using video)
NUM_CONTEXT_IMAGES = int(os.environ.get("NUM_CONTEXT_IMAGES", 16))  # Number of context images for video
CAP_NUM_FRAMES = int(os.environ.get("CAP_NUM_FRAMES", 4))

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
DEFAULT_BOX_START_TOKEN = "<box_start>"
DEFAULT_BOX_END_TOKEN = "<box_end>"
