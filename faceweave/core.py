import os

os.environ['OMP_NUM_THREADS'] = '1'

import signal
import ssl
import sys
import warnings
import platform
import shutil
import onnxruntime
from argparse import ArgumentParser, HelpFormatter

import faceweave.choices
import faceweave.globals
from faceweave.face_analyser import get_one_face, get_average_face
from faceweave.face_store import get_reference_faces, append_reference_face
from faceweave.vision import get_video_frame, detect_fps, read_image, read_static_images
from faceweave import face_analyser, face_masker, content_analyser, metadata, logger, wording
from faceweave.content_analyser import analyse_image, analyse_video
from faceweave.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from faceweave.common_helper import create_metavar
from faceweave.execution_helper import encode_execution_providers, decode_execution_providers
from faceweave.normalizer import normalize_output_path, normalize_padding
from faceweave.filesystem import is_image, is_video, list_module_names, get_temp_frame_paths, create_temp, move_temp, clear_temp
from faceweave.ffmpeg import extract_frames, compress_image, merge_video, restore_audio

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')

if platform.system().lower() == 'darwin':
	ssl._create_default_https_context = ssl._create_unverified_context


def cli() -> None:
	signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
	program = ArgumentParser(formatter_class = lambda prog: HelpFormatter(prog, max_help_position = 120), add_help = False)
	# general
	program.add_argument('-s', '--source', action = 'append', help = wording.get('source_help'), dest = 'source_paths')
	program.add_argument('-t', '--target', help = wording.get('target_help'), dest = 'target_path')
	program.add_argument('-o', '--output', help = wording.get('output_help'), dest = 'output_path')
	program.add_argument('-v', '--version', version = metadata.get('name') + ' ' + metadata.get('version'), action = 'version')
	# misc
	group_misc = program.add_argument_group('misc')
	group_misc.add_argument('--skip-download', help = wording.get('skip_download_help'), action = 'store_true')
	group_misc.add_argument('--headless', help = wording.get('headless_help'), action = 'store_true')
	group_misc.add_argument('--log-level', help = wording.get('log_level_help'), default = 'info', choices = logger.get_log_levels())
	# execution
	execution_providers = encode_execution_providers(onnxruntime.get_available_providers())
	group_execution = program.add_argument_group('execution')
	group_execution.add_argument('--execution-providers', help = wording.get('execution_providers_help').format(choices = ', '.join(execution_providers)), default = [ 'cpu' ], choices = execution_providers, nargs = '+', metavar = 'EXECUTION_PROVIDERS')
	group_execution.add_argument('--execution-thread-count', help = wording.get('execution_thread_count_help'), type = int, default = 4, choices = faceweave.choices.execution_thread_count_range, metavar = create_metavar(faceweave.choices.execution_thread_count_range))
	group_execution.add_argument('--execution-queue-count', help = wording.get('execution_queue_count_help'), type = int, default = 1, choices = faceweave.choices.execution_queue_count_range, metavar = create_metavar(faceweave.choices.execution_queue_count_range))
	group_execution.add_argument('--max-memory', help = wording.get('max_memory_help'), type = int, choices = faceweave.choices.max_memory_range, metavar = create_metavar(faceweave.choices.max_memory_range))
	# face analyser
	group_face_analyser = program.add_argument_group('face analyser')
	group_face_analyser.add_argument('--face-analyser-order', help = wording.get('face_analyser_order_help'), default = 'left-right', choices = faceweave.choices.face_analyser_orders)
	group_face_analyser.add_argument('--face-analyser-age', help = wording.get('face_analyser_age_help'), choices = faceweave.choices.face_analyser_ages)
	group_face_analyser.add_argument('--face-analyser-gender', help = wording.get('face_analyser_gender_help'), choices = faceweave.choices.face_analyser_genders)
	group_face_analyser.add_argument('--face-detector-model', help = wording.get('face_detector_model_help'), default = 'retinaface', choices = faceweave.choices.face_detector_models)
	group_face_analyser.add_argument('--face-detector-size', help = wording.get('face_detector_size_help'), default = '640x640', choices = faceweave.choices.face_detector_sizes)
	group_face_analyser.add_argument('--face-detector-score', help = wording.get('face_detector_score_help'), type = float, default = 0.5, choices = faceweave.choices.face_detector_score_range, metavar = create_metavar(faceweave.choices.face_detector_score_range))
	# face selector
	group_face_selector = program.add_argument_group('face selector')
	group_face_selector.add_argument('--face-selector-mode', help = wording.get('face_selector_mode_help'), default = 'reference', choices = faceweave.choices.face_selector_modes)
	group_face_selector.add_argument('--reference-face-position', help = wording.get('reference_face_position_help'), type = int, default = 0)
	group_face_selector.add_argument('--reference-face-distance', help = wording.get('reference_face_distance_help'), type = float, default = 0.6, choices = faceweave.choices.reference_face_distance_range, metavar = create_metavar(faceweave.choices.reference_face_distance_range))
	group_face_selector.add_argument('--reference-frame-number', help = wording.get('reference_frame_number_help'), type = int, default = 0)
	# face mask
	group_face_mask = program.add_argument_group('face mask')
	group_face_mask.add_argument('--face-mask-types', help = wording.get('face_mask_types_help').format(choices = ', '.join(faceweave.choices.face_mask_types)), default = [ 'box' ], choices = faceweave.choices.face_mask_types, nargs = '+', metavar = 'FACE_MASK_TYPES')
	group_face_mask.add_argument('--face-mask-blur', help = wording.get('face_mask_blur_help'), type = float, default = 0.3, choices = faceweave.choices.face_mask_blur_range, metavar = create_metavar(faceweave.choices.face_mask_blur_range))
	group_face_mask.add_argument('--face-mask-padding', help = wording.get('face_mask_padding_help'), type = int, default = [ 0, 0, 0, 0 ], nargs = '+')
	group_face_mask.add_argument('--face-mask-regions', help = wording.get('face_mask_regions_help').format(choices = ', '.join(faceweave.choices.face_mask_regions)), default = faceweave.choices.face_mask_regions, choices = faceweave.choices.face_mask_regions,  nargs = '+', metavar = 'FACE_MASK_REGIONS')
	# frame extraction
	group_frame_extraction = program.add_argument_group('frame extraction')
	group_frame_extraction.add_argument('--trim-frame-start', help = wording.get('trim_frame_start_help'), type = int)
	group_frame_extraction.add_argument('--trim-frame-end', help = wording.get('trim_frame_end_help'), type = int)
	group_frame_extraction.add_argument('--temp-frame-format', help = wording.get('temp_frame_format_help'), default = 'jpg', choices = faceweave.choices.temp_frame_formats)
	group_frame_extraction.add_argument('--temp-frame-quality', help = wording.get('temp_frame_quality_help'), type = int, default = 100, choices = faceweave.choices.temp_frame_quality_range, metavar = create_metavar(faceweave.choices.temp_frame_quality_range))
	group_frame_extraction.add_argument('--keep-temp', help = wording.get('keep_temp_help'), action = 'store_true')
	# output creation
	group_output_creation = program.add_argument_group('output creation')
	group_output_creation.add_argument('--output-image-quality', help = wording.get('output_image_quality_help'), type = int, default = 80, choices = faceweave.choices.output_image_quality_range, metavar = create_metavar(faceweave.choices.output_image_quality_range))
	group_output_creation.add_argument('--output-video-encoder', help = wording.get('output_video_encoder_help'), default = 'libx264', choices = faceweave.choices.output_video_encoders)
	group_output_creation.add_argument('--output-video-quality', help = wording.get('output_video_quality_help'), type = int, default = 80, choices = faceweave.choices.output_video_quality_range, metavar = create_metavar(faceweave.choices.output_video_quality_range))
	group_output_creation.add_argument('--keep-fps', help = wording.get('keep_fps_help'), action = 'store_true')
	group_output_creation.add_argument('--nsfw', help = wording.get('nsfw_help'), dest='nsfw', action='store_true')
	group_output_creation.add_argument('--skip-audio', help = wording.get('skip_audio_help'), action = 'store_true')
	# frame processors
	available_frame_processors = list_module_names('faceweave/processors/frame/modules')
	program = ArgumentParser(parents = [ program ], formatter_class = program.formatter_class, add_help = True)
	group_frame_processors = program.add_argument_group('frame processors')
	group_frame_processors.add_argument('--frame-processors', help = wording.get('frame_processors_help').format(choices = ', '.join(available_frame_processors)), default = [ 'face_swapper' ], nargs = '+')
	for frame_processor in available_frame_processors:
		frame_processor_module = load_frame_processor_module(frame_processor)
		frame_processor_module.register_args(group_frame_processors)
	# uis
	group_uis = program.add_argument_group('uis')
	group_uis.add_argument('--ui-layouts', help = wording.get('ui_layouts_help').format(choices = ', '.join(list_module_names('faceweave/uis/layouts'))), default = [ 'default' ], nargs = '+')
	run(program)


def apply_args(program : ArgumentParser) -> None:
	args = program.parse_args()
	# general
	faceweave.globals.source_paths = args.source_paths
	faceweave.globals.target_path = args.target_path
	faceweave.globals.output_path = normalize_output_path(faceweave.globals.source_paths, faceweave.globals.target_path, args.output_path)
	# misc
	faceweave.globals.skip_download = args.skip_download
	faceweave.globals.headless = args.headless
	faceweave.globals.log_level = args.log_level
	# execution
	faceweave.globals.execution_providers = decode_execution_providers(args.execution_providers)
	faceweave.globals.execution_thread_count = args.execution_thread_count
	faceweave.globals.execution_queue_count = args.execution_queue_count
	faceweave.globals.max_memory = args.max_memory
	# face analyser
	faceweave.globals.face_analyser_order = args.face_analyser_order
	faceweave.globals.face_analyser_age = args.face_analyser_age
	faceweave.globals.face_analyser_gender = args.face_analyser_gender
	faceweave.globals.face_detector_model = args.face_detector_model
	faceweave.globals.face_detector_size = args.face_detector_size
	faceweave.globals.face_detector_score = args.face_detector_score
	# face selector
	faceweave.globals.face_selector_mode = args.face_selector_mode
	faceweave.globals.reference_face_position = args.reference_face_position
	faceweave.globals.reference_face_distance = args.reference_face_distance
	faceweave.globals.reference_frame_number = args.reference_frame_number
	# face mask
	faceweave.globals.face_mask_types = args.face_mask_types
	faceweave.globals.face_mask_blur = args.face_mask_blur
	faceweave.globals.face_mask_padding = normalize_padding(args.face_mask_padding)
	faceweave.globals.face_mask_regions = args.face_mask_regions
	# frame extraction
	faceweave.globals.trim_frame_start = args.trim_frame_start
	faceweave.globals.trim_frame_end = args.trim_frame_end
	faceweave.globals.temp_frame_format = args.temp_frame_format
	faceweave.globals.temp_frame_quality = args.temp_frame_quality
	faceweave.globals.keep_temp = args.keep_temp
	# output creation
	faceweave.globals.output_image_quality = args.output_image_quality
	faceweave.globals.output_video_encoder = args.output_video_encoder
	faceweave.globals.output_video_quality = args.output_video_quality
	faceweave.globals.keep_fps = args.keep_fps
	faceweave.globals.nsfw = args.nsfw
	faceweave.globals.skip_audio = args.skip_audio
	# frame processors
	available_frame_processors = list_module_names('faceweave/processors/frame/modules')
	faceweave.globals.frame_processors = args.frame_processors
	for frame_processor in available_frame_processors:
		frame_processor_module = load_frame_processor_module(frame_processor)
		frame_processor_module.apply_args(program)
	# uis
	faceweave.globals.ui_layouts = args.ui_layouts


def run(program : ArgumentParser) -> None:
	apply_args(program)
	logger.init(faceweave.globals.log_level)
	limit_resources()
	if not pre_check() or not content_analyser.pre_check() or not face_analyser.pre_check() or not face_masker.pre_check():
		return
	for frame_processor_module in get_frame_processors_modules(faceweave.globals.frame_processors):
		if not frame_processor_module.pre_check():
			return
	if faceweave.globals.headless:
		conditional_process()
	else:
		import faceweave.uis.core as ui

		for ui_layout in ui.get_ui_layouts_modules(faceweave.globals.ui_layouts):
			if not ui_layout.pre_check():
				return
		ui.launch()


def destroy() -> None:
	if faceweave.globals.target_path:
		clear_temp(faceweave.globals.target_path)
	sys.exit()


def limit_resources() -> None:
	if faceweave.globals.max_memory:
		memory = faceweave.globals.max_memory * 1024 ** 3
		if platform.system().lower() == 'darwin':
			memory = faceweave.globals.max_memory * 1024 ** 6
		if platform.system().lower() == 'windows':
			import ctypes

			kernel32 = ctypes.windll.kernel32 # type: ignore[attr-defined]
			kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
		else:
			import resource

			resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
	if sys.version_info < (3, 9):
		logger.error(wording.get('python_not_supported').format(version = '3.9'), __name__.upper())
		return False
	if not shutil.which('ffmpeg'):
		logger.error(wording.get('ffmpeg_not_installed'), __name__.upper())
		return False
	return True


def conditional_process() -> None:
	conditional_append_reference_faces()
	for frame_processor_module in get_frame_processors_modules(faceweave.globals.frame_processors):
		if not frame_processor_module.pre_process('output'):
			return
	if is_image(faceweave.globals.target_path):
		process_image()
	if is_video(faceweave.globals.target_path):
		process_video()


def conditional_append_reference_faces() -> None:
	if 'reference' in faceweave.globals.face_selector_mode and not get_reference_faces():
		source_frames = read_static_images(faceweave.globals.source_paths)
		source_face = get_average_face(source_frames)
		if is_video(faceweave.globals.target_path):
			reference_frame = get_video_frame(faceweave.globals.target_path, faceweave.globals.reference_frame_number)
		else:
			reference_frame = read_image(faceweave.globals.target_path)
		reference_face = get_one_face(reference_frame, faceweave.globals.reference_face_position)
		append_reference_face('origin', reference_face)
		if source_face and reference_face:
			for frame_processor_module in get_frame_processors_modules(faceweave.globals.frame_processors):
				reference_frame = frame_processor_module.get_reference_frame(source_face, reference_face, reference_frame)
				reference_face = get_one_face(reference_frame, faceweave.globals.reference_face_position)
				append_reference_face(frame_processor_module.__name__, reference_face)


def process_image() -> None:
	if analyse_image(faceweave.globals.target_path):
		if faceweave.globals.nsfw:
			pass
		else:
			exit()
	shutil.copy2(faceweave.globals.target_path, faceweave.globals.output_path)
	# process frame
	for frame_processor_module in get_frame_processors_modules(faceweave.globals.frame_processors):
		logger.info(wording.get('processing'), frame_processor_module.NAME)
		frame_processor_module.process_image(faceweave.globals.source_paths, faceweave.globals.output_path, faceweave.globals.output_path)
		frame_processor_module.post_process()
	# compress image
	logger.info(wording.get('compressing_image'), __name__.upper())
	if not compress_image(faceweave.globals.output_path):
		logger.error(wording.get('compressing_image_failed'), __name__.upper())
	# validate image
	if is_image(faceweave.globals.output_path):
		logger.info(wording.get('processing_image_succeed'), __name__.upper())
	else:
		logger.error(wording.get('processing_image_failed'), __name__.upper())


def process_video() -> None:
	if analyse_video(faceweave.globals.target_path, faceweave.globals.trim_frame_start, faceweave.globals.trim_frame_end):
		if faceweave.globals.nsfw:
			pass
		else:
			exit()
	fps = detect_fps(faceweave.globals.target_path) if faceweave.globals.keep_fps else 25.0
	# create temp
	logger.info(wording.get('creating_temp'), __name__.upper())
	create_temp(faceweave.globals.target_path)
	# extract frames
	logger.info(wording.get('extracting_frames_fps').format(fps = fps), __name__.upper())
	extract_frames(faceweave.globals.target_path, fps)
	# process frame
	temp_frame_paths = get_temp_frame_paths(faceweave.globals.target_path)
	if temp_frame_paths:
		for frame_processor_module in get_frame_processors_modules(faceweave.globals.frame_processors):
			logger.info(wording.get('processing'), frame_processor_module.NAME)
			frame_processor_module.process_video(faceweave.globals.source_paths, temp_frame_paths)
			frame_processor_module.post_process()
	else:
		logger.error(wording.get('temp_frames_not_found'), __name__.upper())
		return
	# merge video
	logger.info(wording.get('merging_video_fps').format(fps = fps), __name__.upper())
	if not merge_video(faceweave.globals.target_path, fps):
		logger.error(wording.get('merging_video_failed'), __name__.upper())
		return
	# handle audio
	if faceweave.globals.skip_audio:
		logger.info(wording.get('skipping_audio'), __name__.upper())
		move_temp(faceweave.globals.target_path, faceweave.globals.output_path)
	else:
		logger.info(wording.get('restoring_audio'), __name__.upper())
		if not restore_audio(faceweave.globals.target_path, faceweave.globals.output_path):
			logger.warn(wording.get('restoring_audio_skipped'), __name__.upper())
			move_temp(faceweave.globals.target_path, faceweave.globals.output_path)
	# clear temp
	logger.info(wording.get('clearing_temp'), __name__.upper())
	clear_temp(faceweave.globals.target_path)
	# validate video
	if is_video(faceweave.globals.output_path):
		logger.info(wording.get('processing_video_succeed'), __name__.upper())
	else:
		logger.error(wording.get('processing_video_failed'), __name__.upper())
