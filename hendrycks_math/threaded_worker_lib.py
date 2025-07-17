import transformers
import queue
import threading
from typing import Callable, Optional, Iterator
from hendrycks_math.solve_and_eval_lib import solve_problem
from hendrycks_math.utils import generation_utils
from hendrycks_math.utils import math_types
from hendrycks_math.utils.math_types import GradedSolutionAttempt, ProblemQueueEntry
from hendrycks_math.solution_builder import GenerationBuilder

SENTINEL_PROBLEM_ID = -10
SentinelProblem = ProblemQueueEntry(
    id=SENTINEL_PROBLEM_ID,
    problem="Sentinel problem",
    reference_answer="Sentinel answer",
    reference_solution="Sentinel solution",
)
SentinelGradedSolutionAttempt = GradedSolutionAttempt(
    problem_id=SENTINEL_PROBLEM_ID,
    solution_attempt=None, # type: ignore
    is_correct=False,
    reference_answer="Sentinel answer",
)


class IfgSamplerWorker(threading.Thread):
    """A worker thread that that samples IFG solutions and graded them.

    Each worker is responsible for solving a single problem at a time.
    Setting max_n_steps to 1 will make the worker equivalent to vanilla sampling.
    """

    def __init__(
        self,
        problem_queue: queue.Queue[ProblemQueueEntry],
        solution_queue: queue.Queue[GradedSolutionAttempt],
        model: math_types.Model,
        model_name: str,
        model_type: str,
        model_tokenizer: Optional[transformers.AutoTokenizer],
        prompt_template: str | math_types.JsonMessages,
        max_tokens_per_step: int,
        temperature_even_index: float,
        temperature_odd_index: Optional[float],
        max_n_steps: int,
        solution_end: str,
        step_separator: str,
        evaluation_fn: Callable[..., bool],
    ):
        super().__init__()
        self.problem_queue = problem_queue
        self.solution_queue = solution_queue
        self.model = model
        self.evaluation_fn = evaluation_fn
        self.solution_end = solution_end
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature_even_index = temperature_even_index
        self.temperature_odd_index = temperature_odd_index
        self.max_n_steps = max_n_steps
        self.step_separator = step_separator
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.daemon = (
            True  # Allow the main thread to exit even if this thread is still running
        )
        self.solution_builder_factory = (
            GenerationBuilder.get_generation_builder_factory(
                prompt_template=prompt_template,
                model_type=model_type,
                tokenizer=model_tokenizer,
            )
        )

    def run(self):
        """Continuously consume and solve problems until sentinel is seen.

        The thread exits cleanly after pushing a sentinel result onto the
        solution queue so the iterator can detect that this worker has
        finished.  This avoids the need for the master thread to poll
        `thread.is_alive()` or `queue.qsize()`.
        """

        while True:
            problem_entry = self.problem_queue.get()

            # Sentinel → worker shutdown
            if problem_entry.id == SENTINEL_PROBLEM_ID:
                # Notify the iterator that this worker has finished.
                self.solution_queue.put(SentinelGradedSolutionAttempt)
                break

            # Normal work item
            solution_attempt = self.solution_builder_factory(problem_entry.problem)
            n_shots = solution_attempt.prompt_string().count(self.solution_end)

            solution_attempt = solve_problem(
                solution_attempt=solution_attempt,
                model=self.model,
                n_shots=n_shots,
                max_steps=self.max_n_steps,
                temperature_even_index=self.temperature_even_index,
                temperature_odd_index=self.temperature_odd_index,
                solution_end_marker=self.solution_end,
                get_sampling_params=self.get_sampling_params,
            )

            is_correct = self.evaluation_fn(
                solution_attempt=solution_attempt, problem_definition=problem_entry
            )

            graded = GradedSolutionAttempt(
                problem_id=problem_entry.id,
                solution_attempt=solution_attempt,
                is_correct=is_correct,
                reference_answer=problem_entry.reference_answer,
            )
            self.solution_queue.put(graded)

    def get_sampling_params(self, temperature: float):
        return generation_utils.get_sampling_params_for_model(
            self.model,
            temperature,
            self.max_tokens_per_step,
            n=1,
            stop_str=self.step_separator,
            include_stop_str_in_output=True,
            model_name=self.model_name,
        )


class MultiThreadedIFGSampler:
    """Run n_workers parallel workers to solve problems using IFG."""

    def __init__(
        self,
        n_workers: int,
        problem_queue: queue.Queue,
        model: math_types.Model,
        model_name: str,
        model_type: str,
        model_tokenizer: Optional[transformers.AutoTokenizer],
        prompt_template: str,
        max_tokens_per_step: int,
        temperature_even_index: float,
        temperature_odd_index: Optional[float],
        max_n_steps: int,
        solution_end: str,
        step_separator: str,
        evaluation_fn: Callable[..., bool],
    ):
        self.worker_threads = []
        self.problem_queue = problem_queue
        self._solution_queue: queue.Queue[GradedSolutionAttempt] = queue.Queue()

        for _ in range(n_workers):
            worker = IfgSamplerWorker(
                problem_queue=self.problem_queue,
                solution_queue=self._solution_queue,
                model=model,
                model_name=model_name,
                model_type=model_type,
                model_tokenizer=model_tokenizer,
                prompt_template=prompt_template,
                max_tokens_per_step=max_tokens_per_step,
                temperature_even_index=temperature_even_index,
                temperature_odd_index=temperature_odd_index,
                max_n_steps=max_n_steps,
                solution_end=solution_end,
                step_separator=step_separator,
                evaluation_fn=evaluation_fn,
            )
            self.worker_threads.append(worker)

        # Insert one sentinel problem for each worker so every thread knows
        # when to terminate once the real workload is exhausted.
        for _ in range(len(self.worker_threads)):
            self.problem_queue.put(SentinelProblem)

    def start(self):
        for worker in self.worker_threads:
            worker.start()

    def __iter__(self) -> Iterator[GradedSolutionAttempt]:
        """Yield graded attempts as they arrive until all workers finish.

        We block on `queue.get()` instead of polling `qsize()`.  Each worker
        pushes a *sentinel* `GradedSolutionAttempt` after it processes its
        final job, so the iterator can count how many workers have finished
        without relying on non-atomic size hints.
        """

        remaining_sentinels = len(self.worker_threads)

        while remaining_sentinels > 0:
            result = self._solution_queue.get()

            if result.problem_id == SENTINEL_PROBLEM_ID:
                remaining_sentinels -= 1
                continue  # Don't yield sentinel to caller

            yield result

    def close(self):
        empty_queue(self._solution_queue)
        empty_queue(self.problem_queue)
        for thread in self.worker_threads:
            thread.join()
        self.worker_threads = [
            thread for thread in self.worker_threads if thread.is_alive()
        ]
        assert len(self.worker_threads) == 0


def empty_queue(queue: queue.Queue):
    """Remove all elements from the queue."""
    while not queue.empty():
        queue.get()
