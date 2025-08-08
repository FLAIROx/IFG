"""Lib for generating and evaluating math problems.

Samples solutions to maths problems using multi-step IFG.
Also provides unfified signature
wrappers for evaluating MATH and Omni-MATH problems."""

import abc
import copy
from typing import Callable, Optional
from uuid import uuid5

import transformers

import gllm
from hendrycks_math import math_eval
from hendrycks_math.utils import generation_utils
from hendrycks_math.utils.math_types import (
    BaseModels,
    ProblemQueueEntry,
    ProblemSolutionRecord,
)
from hendrycks_math.solution_builder import GenerationBuilder
from hendrycks_math.utils import math_types


def solve_problem(
    solution_attempt: GenerationBuilder,
    model: math_types.Model,
    n_shots: int,
    max_steps: int,
    temperature_even_index: float,
    temperature_odd_index: Optional[float],
    solution_end_marker: str,
    get_sampling_params: Callable[[float], math_types.GenericSamplingParams],
) -> ProblemSolutionRecord:
    """Sample solution to a math problem using a multi-step IFG.

    Repeated samples with alternating temperatures until the solution is terminated.
    If n_steps=1 then it is equivalent to vanilla sampling.

    Args:
    problem_prompt (GenerationBuilder): A GenerationBuilder preseeded with the problem and
        necessary prompt context.
    model (math_types.Model): The language model to use.
    n_shots (int): The number of few-shot examples in the prompt.
    max_steps (int): The maximum number of steps to take.
    temperature_even_index (float): The temperature for even steps. Although
        prompt dependent, typically this will be the intent step.
    temperature_odd_index (float): The temperature for odd steps. Although
        prompt dependent, typically this will be the generation step.
    solution_end_marker (str): The string that denotes the end of a solution.
    get_sampling_params (Callable[[float], math_types.GenericSamplingParams]): A function
        that returns the sampling parameters for a given temperature.

    Returns:
    Optional[ProblemSolutionRecord]: The solution record. Contains
        the problem, the prompt, the solution attempt, and whether the solution
        was terminated.
    """
    conversation_id = str(uuid5(uuid5.NAMESPACE_DNS, str(uuid5.uuid4())))
    terminated = False
    result = None
    for j in range(max_steps):
        temperature = temperature_even_index if j % 2 == 0 else temperature_odd_index
        if temperature is None:
            if j % 2 == 0:
                raise ValueError("temperature_even_index must be set if max_steps > 0")
            else:
                raise ValueError("temperature_odd_index must be set if max_steps > 1")

        sampling_params = get_sampling_params(temperature)
        response = generation_utils.generate_from_model(
            model, [solution_attempt.prompt], sampling_params,  # type: ignore
            conversation_id=conversation_id
        )[0]

        solution_attempt.add_step(response)

        terminated = is_terminated(solution_attempt, n_shots, solution_end_marker)
        prompt_steps = solution_attempt.n_initial_steps
        result = ProblemSolutionRecord(
            prompt=solution_attempt.prompt_string(start_step=0, end_step=prompt_steps),
            solution_attempt=solution_attempt.prompt_string(start_step=prompt_steps),
            terminated=terminated,
        )
        if terminated:
            return result

    assert result is not None, "max_steps must be greater than 0"
    return result


def is_terminated(solution: GenerationBuilder, n_shots: int, solution_end: str) -> bool:
    """Check if the end of solution sequence is reached.

    Args:
        solution (GenerationBuilder): The solution attempt.
        n_shots (int): The number of few-shot examples in the prompt.
        solution_end (str): The string that denotes the end of a solution.
    """
    text = solution.prompt_string()
    return text.count(solution_end) > n_shots


def is_correct_math(
    solution_attempt: ProblemSolutionRecord, problem_definition: ProblemQueueEntry
) -> bool:
    """Evaluate the correctness of MATH (Hendrycks et al.) solution attempt."""
    return math_eval.compare_final_answer_to_ground_truth(
        solution_attempt.solution_attempt, problem_definition.reference_answer
    )


def is_correct_mmlu(
    solution_attempt: ProblemSolutionRecord,
    problem_definition: ProblemQueueEntry,
    filler: str,
) -> bool:
    """
    Check if the response is correct.
    """

    response = solution_attempt.solution_attempt
    # Remove any leading or trailing whitespace
    response = response.strip()
    response = response.split("\n")

    # Find the last line of the response that is not whitespace
    candidate_answer = None
    for line in reversed(response):
        if line.strip():
            if line.strip() == filler:
                continue
            candidate_answer = line.strip()[0]
            break
    return candidate_answer == problem_definition.reference_answer
