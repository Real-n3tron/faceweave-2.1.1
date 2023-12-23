from typing import Any, Dict
from functools import lru_cache
import threading
import cv2
import numpy
import onnxruntime
from tqdm import tqdm

import faceweave.globals
from faceweave import wording
from faceweave.typing import Frame, ModelValue
from faceweave.vision import get_video_frame, count_video_frame_total, read_image, detect_fps
from faceweave.filesystem import resolve_relative_path
from faceweave.download import conditional_download

CONTENT_ANALYSER = None
THREAD_LOCK : threading.Lock = threading.Lock()
MODELS : Dict[str, ModelValue] =\
{
	'open_nsfw':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/open_nsfw.onnx',
		'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
	}
}
MAX_PROBABILITY = 0.80
MAX_RATE = 5
STREAM_COUNTER = 0


def get_content_analyser() -> Any:
	global CONTENT_ANALYSER

	with THREAD_LOCK:
		if CONTENT_ANALYSER is None:
			model_path = MODELS.get('open_nsfw').get('path')
			CONTENT_ANALYSER = onnxruntime.InferenceSession(model_path, providers = faceweave.globals.execution_providers)
	return CONTENT_ANALYSER


def clear_content_analyser() -> None:
	global CONTENT_ANALYSER

	CONTENT_ANALYSER = None


def pre_check() -> bool:
	if not faceweave.globals.skip_download:
		download_directory_path = resolve_relative_path('../.assets/models')
		model_url = MODELS.get('open_nsfw').get('url')
		conditional_download(download_directory_path, [ model_url ])
	return True


def analyse_stream(frame : Frame, fps : float) -> bool:
	global STREAM_COUNTER

	STREAM_COUNTER = STREAM_COUNTER + 1
	if STREAM_COUNTER % int(fps) == 0:
		return analyse_frame(frame)
	return False


def prepare_frame(frame : Frame) -> Frame:
	frame = cv2.resize(frame, (224, 224)).astype(numpy.float32)
	frame -= numpy.array([ 104, 117, 123 ]).astype(numpy.float32)
	frame = numpy.expand_dims(frame, axis = 0)
	return frame


def analyse_frame(frame : Frame) -> bool:
	return False


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
	
	return False 


@lru_cache(maxsize = None)
def analyse_video(video_path : str, start_frame : int, end_frame : int) -> bool:
	
	return False 
