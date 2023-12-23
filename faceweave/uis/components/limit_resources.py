from typing import Optional
import gradio

import faceweave.globals
import faceweave.choices
from faceweave import wording

MAX_MEMORY_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global MAX_MEMORY_SLIDER

	MAX_MEMORY_SLIDER = gradio.Slider(
		label = wording.get('max_memory_slider_label'),
		step = faceweave.choices.max_memory_range[1] - faceweave.choices.max_memory_range[0],
		minimum = faceweave.choices.max_memory_range[0],
		maximum = faceweave.choices.max_memory_range[-1]
	)


def listen() -> None:
	MAX_MEMORY_SLIDER.change(update_max_memory, inputs = MAX_MEMORY_SLIDER)


def update_max_memory(max_memory : int) -> None:
	faceweave.globals.max_memory = max_memory if max_memory > 0 else None
