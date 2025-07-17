"""Utilities for building solutions from over multiple generation steps.

Abstracts away the details of chat vs non-chat models."""

import abc
from hendrycks_math.utils import math_types
from hendrycks_math.utils.math_types import ModelTypes
import copy
import transformers
from typing import Optional, Callable


class GenerationBuilder(abc.ABC):
    """Handles the step-wise construction of a generation.

    Each step handles the addition of a single phase of the generation.
    This object handles the composition of different phases into a single
    prompt object.
    """

    @staticmethod
    def create(
        prompt_template: str | math_types.JsonMessages,
        model_type: str,
        problem: str,
        tokenizer: Optional[transformers.AutoTokenizer] = None,
    ) -> "GenerationBuilder":
        """Create a suitable GenerationBuilder for the given model and template."""
        if model_type == ModelTypes.BASE:
            assert isinstance(prompt_template, str)
            return NonChatGenerationBuilder(prompt_template, problem)

        elif model_type == ModelTypes.MATHSTRAL:
            assert isinstance(prompt_template, str)
            assert tokenizer is not None
            return MathstralGenerationBuilder(prompt_template, problem, tokenizer)

        elif model_type == ModelTypes.CHAT:
            assert isinstance(prompt_template, list)
            return ChatGenerationBuilder(prompt_template, problem)

        raise ValueError(
            f"Invalid model_type <{model_type}>, "
            f"valid choices are {ModelTypes.__members__.keys()}"
        )

    @staticmethod
    def get_generation_builder_factory(
        prompt_template: str | math_types.JsonMessages,
        model_type: str,
        tokenizer: Optional[transformers.AutoTokenizer] = None,
    ) -> Callable[[str], "GenerationBuilder"]:
        """Return a function that takes a problem and returns a GenerationBuilder."""
        return lambda problem: GenerationBuilder.create(
            prompt_template, model_type, problem, tokenizer
        )

    @property
    @abc.abstractmethod
    def prompt(self) -> str | math_types.JsonMessages:
        """A prompt ready for use in generation."""

    @property
    @abc.abstractmethod
    def n_initial_steps(self) -> int:
        """The number of stpes in inital the prompt"""

    @abc.abstractmethod
    def prompt_string(self, start_step: int = 0, end_step: Optional[int] = None) -> str:
        """A string representation of the prompt.

        If start_step or end_step are provided then only the
        steps between those indices will be included.
        """

    @abc.abstractmethod
    def add_step(self, step: str) -> None:
        """Add a new step"""


class NonChatGenerationBuilder(GenerationBuilder):
    def __init__(
        self,
        prompt_template: str,
        problem: str,
    ) -> None:
        super().__init__()
        self._steps = [prompt_template.format(question=problem)]
        self._problem = problem

    @property
    def prompt(self) -> str:
        return "".join(self._steps)

    @property
    def n_initial_steps(self) -> int:
        return 1

    def prompt_string(self, start_step: int = 0, end_step: Optional[int] = None) -> str:
        if end_step is None:
            end_step = len(self._steps)

        return "".join(self._steps[start_step:end_step])

    def add_step(self, step: str) -> None:
        self._steps.append(step)

    def __str__(self) -> str:
        return f"NonChatGenerationBuilder(problem={self._problem}, steps={len(self._steps)-1})"


class MathstralGenerationBuilder(NonChatGenerationBuilder):
    def __init__(
        self, prompt_template: str, problem: str, tokenizer: transformers.AutoTokenizer
    ) -> None:
        super().__init__(prompt_template, problem)
        # We will inject the question and a dummy solution into the chat templated discussion.
        # We will then remove the dummy solution, along with any end of turn indicators.
        anchor = "*BNl&^(MHI86-"
        assert tokenizer is not None
        messages = [
            {"role": "user", "content": prompt_template.format(question=problem)},
            {"role": "assistant", "content": anchor},
        ]

        context = tokenizer.apply_chat_template(  # type: ignore
            messages,
            tokenize=False,
        )
        self._steps[0] = context[: context.index(anchor)]


class ChatGenerationBuilder(GenerationBuilder):
    def __init__(
        self,
        prompt_template: math_types.JsonMessages,
        problem: str,
    ) -> None:
        super().__init__()
        self._steps = copy.deepcopy(prompt_template)

        # Format the problem into the prompt
        for step in self._steps:
            step["content"] = step["content"].format(question=problem)

        self.n_prompt_turns = len(self._steps)
        self._problem = problem

    @property
    def prompt(self) -> math_types.JsonMessages:
        return self._steps

    @property
    def n_initial_steps(self) -> int:
        return self.n_prompt_turns

    def prompt_string(self, start_step: int = 0, end_step: Optional[int] = None) -> str:
        if end_step is None:
            end_step = len(self._steps)
        steps_to_parse = self._steps[start_step:end_step]
        processed = [x["content"] for x in steps_to_parse if x["role"] == "assistant"]
        return " ".join(processed)
        # return str(steps_to_parse)
        # processed = [f"<role:{step['role']}>\n{step['content']}\n" for step in steps_to_parse]
        # return "".join(processed)

    def add_step(self, step: str) -> None:
        self._steps.append({"role": "assistant", "content": step})
        self._steps.append({"role": "user", "content": "continue"})

    def __str__(self) -> str:
        return f"ChatGenerationBuilder(problem={self._problem}, steps={len(self._steps)-1})"
