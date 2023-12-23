from typing import Optional
import gradio

import faceweave.globals
import faceweave.choices
from faceweave import wording

EXECUTION_QUEUE_COUNT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global EXECUTION_QUEUE_COUNT_SLIDER

	EXECUTION_QUEUE_COUNT_SLIDER = gradio.Slider(
		label = wording.get('execution_queue_count_slider_label'),
		value = faceweave.globals.execution_queue_count,
		step = faceweave.choices.execution_queue_count_range[1] - faceweave.choices.execution_queue_count_range[0],
		minimum = faceweave.choices.execution_queue_count_range[0],
		maximum = faceweave.choices.execution_queue_count_range[-1]
	)


def listen() -> None:
	EXECUTION_QUEUE_COUNT_SLIDER.change(update_execution_queue_count, inputs = EXECUTION_QUEUE_COUNT_SLIDER)


def update_execution_queue_count(execution_queue_count : int = 1) -> None:
	faceweave.globals.execution_queue_count = execution_queue_count
