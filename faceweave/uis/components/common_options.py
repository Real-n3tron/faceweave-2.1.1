from typing import Optional, List
import gradio

import faceweave.globals
from faceweave import wording
from faceweave.uis import choices as uis_choices

COMMON_OPTIONS_CHECKBOX_GROUP : Optional[gradio.Checkboxgroup] = None


def render() -> None:
	global COMMON_OPTIONS_CHECKBOX_GROUP

	value = []
	if faceweave.globals.nsfw:
		value.append('nsfw')
	if faceweave.globals.keep_fps:
		value.append('keep-fps')
	if faceweave.globals.keep_temp:
		value.append('keep-temp')
	if faceweave.globals.skip_audio:
		value.append('skip-audio')
	if faceweave.globals.skip_download:
		value.append('skip-download')
	COMMON_OPTIONS_CHECKBOX_GROUP = gradio.Checkboxgroup(
		label = wording.get('common_options_checkbox_group_label'),
		choices = uis_choices.common_options,
		value = value
	)


def listen() -> None:
	COMMON_OPTIONS_CHECKBOX_GROUP.change(update, inputs = COMMON_OPTIONS_CHECKBOX_GROUP)


def update(common_options : List[str]) -> None:
	faceweave.globals.keep_fps = 'keep-fps' in common_options
	faceweave.globals.keep_temp = 'keep-temp' in common_options
	faceweave.globals.skip_audio = 'skip-audio' in common_options
	faceweave.globals.skip_download = 'skip-download' in common_options
	faceweave.globals.nsfw = 'nsfw' in common_options

