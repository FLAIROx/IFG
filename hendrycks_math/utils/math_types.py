"""Types for solving multi-step IFG math problems."""

import enum
from dataclasses import dataclass
from typing import Optional

import vllm
import gllm
import transformers
import typing
from typing import Any


@dataclass
class ProblemQueueEntry:
    id: int
    problem: str
    reference_answer: str  # Final answer
    reference_solution: str  # Full solution
    prompt: Optional[str] = None  # Fully formatted prompt


@dataclass
class ProblemSolutionRecord:
    prompt: str
    solution_attempt: str
    terminated: bool


@dataclass
class GradedSolutionAttempt:
    problem_id: int
    solution_attempt: ProblemSolutionRecord
    is_correct: bool
    reference_answer: str


class BaseModels(enum.StrEnum):
    MATHSTRAL = "mistralai/Mathstral-7B-v0.1"
    LLAMA_8B = "meta-llama/Llama-3.1-8B"
    LLAMA_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
    GPT4_O_OPENAI = "gpt-4o"


class ModelTypes(enum.StrEnum):
    """Models typed based on prompt formatting."""

    BASE = "base"
    CHAT = "chat"
    # Mathstral is a special case because, after some
    # preprocessing of the prompt we can use it like a base model.
    MATHSTRAL = "mathstral"


class ExternalServingHosts(enum.StrEnum):
    OPENAI = "https://api.openai.com/"
    OPENROUTER = "https://openrouter.ai/api/"


AllExternalHosts = ExternalServingHosts.__members__.values()

GllmGenerationArgs = typing.NewType("GllmGenerationArgs", dict[str, Any])
Model = vllm.LLM | gllm.GLLM | transformers.PreTrainedModel # type: ignore
GenericSamplingParams = (
    vllm.SamplingParams | transformers.GenerationConfig | GllmGenerationArgs # type: ignore
)

JsonMessages = list[dict[str, str]]
