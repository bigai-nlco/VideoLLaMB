CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# # image
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# # video
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vi_patch>"
DEFAULT_VI_START_TOKEN = "<vi_start>"
DEFAULT_VI_END_TOKEN = "<vi_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"

# # X
X_TOKEN_INDEX = {"IMAGE": -200, "VIDEO": -201}
X_INDEX_TOKEN = {-200: "IMAGE", -201: "VIDEO"}
DEFAULT_X_TOKEN = {"IMAGE": "<image>", "VIDEO": "<video>"}
DEFAULT_X_PATCH_TOKEN = {"IMAGE": "<im_patch>", "VIDEO": "<vi_patch>"}
DEFAULT_X_START_TOKEN = {"IMAGE": "<im_start>", "VIDEO": "<vi_start>"}
DEFAULT_X_END_TOKEN = {"IMAGE": "<im_end>", "VIDEO": "<vi_end>"}
X_PLACEHOLDER = {"IMAGE": "<image-placeholder>", "VIDEO": "<video-placeholder>"}

