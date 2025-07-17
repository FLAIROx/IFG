import dataclasses
import math
import time
import json
import logging
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import pydantic
import tyro


class RandomRange(pydantic.BaseModel):
    """Range for continuous float sweep parameter"""

    min: float | int | str
    """Minimum value for the sweep parameter.
    Can be a float, int, or str that contains an inequality.
    with another argument."""
    max: float | int | str


class ParameterSweep(pydantic.BaseModel):
    arg: str
    # Sweeps for continuous float parameters
    random_linear: Optional[RandomRange] = None
    random_log: Optional[RandomRange] = None


class SweepJob(pydantic.BaseModel):
    fixed_args: str
    output_dir_argname: str
    output_dir: str
    best_run_fixed_args: Optional[str] = None
    output_dir_suffix: str  # per run suffix for output dir
    budget: int
    seed: int
    sweep_args: ParameterSweep | list[ParameterSweep]
    results_file: str = "results.json"
    metric: str = "Pass@k"


@dataclasses.dataclass
class Config:
    script: str
    sweep_file: str
    dry_run: bool = False


class SweepedParamNode:
    """A node in the sweep DAG.

    Each node is a parameter to sample, and each edge is a
    constraint dependency i.e. a hparam A > hparam to enforce
    during sampling.
    """

    def __init__(
        self,
        name: str,
        rng: np.random.RandomState,
        lower_bound_scale: float = 1.0,
        lower_bound_offset: float = 0.0,
        upper_bound_scale: float = 1.0,
        upper_bound_offset: float = 0.0,
        lower_bound_parent_name: Optional[str] = None,
        upper_bound_parent_name: Optional[str] = None,
        scale_type: str = "linear",
    ) -> None:
        """Initialize a sweep node."""
        self.lower_bound_scale = lower_bound_scale
        self.lower_bound_offset = lower_bound_offset
        self.upper_bound_scale = upper_bound_scale
        self.upper_bound_offset = upper_bound_offset

        self.lower_bound_parent_name = lower_bound_parent_name
        self.upper_bound_parent_name = upper_bound_parent_name

        if self.lower_bound_parent_name is not None:
            self.lower_bound_parent_value = None
        else:
            self.lower_bound_parent_value = 0

        if self.upper_bound_parent_name is not None:
            self.upper_bound_parent_value = None
        else:
            self.upper_bound_parent_value = 0

        self.name = name

        self.children = set()
        self.ancestors = set()

        self.scale_type = scale_type
        assert self.scale_type in [
            "linear",
            "log",
        ], f"Invalid scale type {self.scale_type}. Must be one of ['linear', 'log']"

        self.rng = rng
        self.sample = None

    def register_sampled_parent(self, parent_name: str, value: float | int):
        """Register the value of parent node that was sampled for all chilren."""
        if parent_name == self.lower_bound_parent_name:
            self.lower_bound_parent_value = value
        if parent_name == self.upper_bound_parent_name:
            self.upper_bound_parent_value = value
        assert parent_name in [
            self.lower_bound_parent_name,
            self.upper_bound_parent_name,
        ], f"Invalid parent name {parent_name}. Must be one of "
        f"[{self.lower_bound_parent_name}, {self.upper_bound_parent_name}]"

    def get_range(self):
        """Get the range of the parameter to sample."""

        assert self.lower_bound_parent_value is not None
        assert self.upper_bound_parent_value is not None

        lower_bound = (
            self.lower_bound_scale * self.lower_bound_parent_value
            + self.lower_bound_offset
        )
        upper_bound = (
            self.upper_bound_scale * self.upper_bound_parent_value
            + self.upper_bound_offset
        )
        return lower_bound, upper_bound

    def trigger_sampling(self):
        lower_bound, upper_bound = self.get_range()
        if self.scale_type == "linear":
            self.sample = np.random.uniform(lower_bound, upper_bound)
        elif self.scale_type == "log":
            lower_bound = math.log(lower_bound, 10)
            upper_bound = math.log(upper_bound, 10)
            self.sample = 10 ** np.random.uniform(lower_bound, upper_bound)

        for child in self.children:
            child.register_sampled_parent(self.name, self.sample)

    def get_sample(self):
        """Get the sampled value."""
        if self.sample is None:
            raise ValueError(f"Node {self.name} has not been sampled yet.")
        return self.sample

    def is_ready_to_sample(self):
        """Check if the node is ready to be sampled.

        A node is ready to be sampled if all its parents have been sampled.
        """
        if (
            self.lower_bound_parent_value is None
            or self.upper_bound_parent_value is None
        ):
            return False
        return True

    def sample_and_propagate(self):
        """Sample the parameter and propagate the sampled value to the children."""
        if not self.is_ready_to_sample():
            return

        self.trigger_sampling()
        for child in self.children:
            child.register_sampled_parent(self.name, self.sample)
            child.sample_and_propagate()

    def add_child(self, child: "SweepedParamNode"):
        self.children.add(child)

    def clear_tree_samples(self):
        self.sample = None
        for child in self.children:
            child.clear_tree_samples()

    def __hash__(self):
        return hash(
            (self.name, self.lower_bound_parent_name, self.upper_bound_parent_name)
        )

    def __repr__(self):
        return (
            f"SweepedParamNode({self.name}, "
            f"> {self.lower_bound_parent_name} * {self.lower_bound_scale} + {self.lower_bound_offset}, "
            f"< {self.upper_bound_parent_name} * {self.upper_bound_scale} + {self.upper_bound_offset})"
        )


class InvalidSweepBoundError(ValueError):
    """Custom error for invalid sweep bounds."""

    def __init__(self, expression: str):
        message = f"Invalid sweep bound expression: {expression}"
        super().__init__(message)


def get_bound(
    min_bound: str | int | float,
    max_bound: str | int | float,
    mode: str,
) -> tuple[str | None, float, float]:
    """Get the bound for a sweep parameter.
    Each bound is an optional parent name, offset, and scale.

    Args:
        min_bound: The minimum bound for the parameter.
        max_bound: The maximum bound for the parameter.
        mode: The mode of the sweep. Can be "upper" or "lower".
    Returns:
        A tuple containing the bound parent,  the offset, and the scale.
    """
    if mode == "upper":
        bound = max_bound
    elif mode == "lower":
        bound = min_bound
    else:
        raise ValueError(f"Invalid mode {mode}. Must be one of ['upper', 'lower']")

    if isinstance(bound, (int, float)):
        return None, bound, 0

    assert isinstance(bound, str)
    operators = "*+"

    if bound.count("*") > 1 or bound.count("+") > 1:
        raise InvalidSweepBoundError(bound)

    def is_numeric(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    offset = 0
    scale = 1
    parent_name = None
    parse_tree = bound.split("+")
    parse_tree = [x.strip() for x in parse_tree]

    if len(parse_tree) > 2:
        raise InvalidSweepBoundError(bound)

    if len(parse_tree) == 1:
        offset = 0
        multiplicative = parse_tree[0]

    elif is_numeric(parse_tree[0]):
        offset = float(parse_tree[0])
        multiplicative = parse_tree[1]

    else:
        multiplicative = parse_tree[0]
        offset = float(parse_tree[1])

    multiplicative = multiplicative.split("*")
    multiplicative = [x.strip() for x in multiplicative]
    if len(multiplicative) > 2:
        raise InvalidSweepBoundError(bound)
    if len(multiplicative) == 1:
        parent_name = multiplicative[0]
        scale = 1

    elif is_numeric(multiplicative[0]):
        scale = float(multiplicative[0])
        parent_name = multiplicative[1]
    else:
        scale = float(multiplicative[1])
        parent_name = multiplicative[0]

    if not isinstance(parent_name, str) or is_numeric(parent_name):
        raise InvalidSweepBoundError(bound)

    return parent_name, offset, scale


class DagCycleError(ValueError):
    """Custom error for cycle detection in the DAG."""

    def __init__(self, nodes_in_cycle: set[SweepedParamNode]):
        self.nodes_in_cycle = nodes_in_cycle
        super().__init__(f"Cycle detected in sweep graph: {nodes_in_cycle}")


def check_for_cycles(nodes: dict[str, SweepedParamNode], roots: list[SweepedParamNode]):
    """Check for cycles in the graph.
    Args:
        nodes: The nodes in the graph.
        roots: The root nodes in the graph.
    Raises:
        CycleError: If a cycle is detected in the graph.
    """

    # Check for directed cycles
    visited = set()
    stack = roots[::]

    while stack:
        node = stack.pop()
        new_ancestors = {node} | node.ancestors

        for child in node.children:
            if child.ancestors.intersection(new_ancestors):
                nodes_in_cycle = child.ancestors.intersection(new_ancestors) | {child}
                raise DagCycleError(nodes_in_cycle)

            child.ancestors = new_ancestors

            if child not in visited:
                stack.append(child)
                visited.add(child)


def validate_sweep_job(sweep_job: SweepJob):
    """Check that the sweep job is valid.

    Make sure that only one search type is set for each sweep arg."""
    if isinstance(sweep_job.sweep_args, ParameterSweep):
        parameter_sweeps = [sweep_job.sweep_args]
    else:
        parameter_sweeps = sweep_job.sweep_args

    invalid_args = []
    for sweep_arg in parameter_sweeps:
        # Only one of these should be set
        if (sweep_arg.random_linear is None) == (sweep_arg.random_log is None):
            invalid_args.append(sweep_arg)

    if invalid_args:
        raise ValueError(
            f"Invalid sweep args: {invalid_args}."
            " Must specify exactly one of random_linear or random_log"
        )


# def sample_value(rng: np.random.RandomState, sweep_arg: ParameterSweep):
#     """Sample a value for a sweep parameter."""
#     if sweep_arg.random_linear:
#         return rng.uniform(sweep_arg.random_linear.min, sweep_arg.random_linear.max)
#     elif sweep_arg.random_log:
#         return 10 ** rng.uniform(
#             np.log10(sweep_arg.random_log.min), np.log10(sweep_arg.random_log.max)
#         )
#     else:
#         raise ValueError("Must specify exactly one of random_linear or random_log")


def get_result(results_file_path: str, metric: str) -> float | int:
    """Get a result from a results file.
    Args:
        results_file_path: Path to the results file.
        metric: Name of the metric to extract
    Returns:
        The value of the metric from the file
    """
    with open(results_file_path, "r") as f:
        results = json.load(f)
    return results[metric]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_sweep_dag(
    sweep_args: list[ParameterSweep], rng: np.random.RandomState
) -> tuple[dict[str, SweepedParamNode], list[SweepedParamNode]]:
    """Build a DAG of sweep parameters.

    Args:
        sweep_args: The sweep args to build the DAG from.
        rng: The random number generator to use for sampling.
    Returns:
        A dictionary of nodes and a list of root nodes in the DAG.
    """
    nodes = {}
    roots = []

    for sweep_arg in sweep_args:
        if sweep_arg.arg not in nodes:
            nodes[sweep_arg.arg] = SweepedParamNode(sweep_arg.arg, rng)

        if sweep_arg.random_linear:
            min_bound = sweep_arg.random_linear.min
            max_bound = sweep_arg.random_linear.max
            lower_bound_parent_name, lower_bound_offset, lower_bound_scale = get_bound(
                min_bound, max_bound, "lower"
            )
            upper_bound_parent_name, upper_bound_offset, upper_bound_scale = get_bound(
                min_bound, max_bound, "upper"
            )
            node = SweepedParamNode(
                sweep_arg.arg,
                rng,
                lower_bound_scale=lower_bound_scale,
                lower_bound_offset=lower_bound_offset,
                upper_bound_scale=upper_bound_scale,
                upper_bound_offset=upper_bound_offset,
                lower_bound_parent_name=lower_bound_parent_name,
                upper_bound_parent_name=upper_bound_parent_name,
                scale_type="linear",
            )

            nodes[sweep_arg.arg] = node

        elif sweep_arg.random_log:
            min_bound = sweep_arg.random_log.min
            max_bound = sweep_arg.random_log.max
            lower_bound_parent_name, lower_bound_offset, lower_bound_scale = get_bound(
                min_bound, max_bound, "lower"
            )
            upper_bound_parent_name, upper_bound_offset, upper_bound_scale = get_bound(
                min_bound, max_bound, "upper"
            )
            node = SweepedParamNode(
                sweep_arg.arg,
                rng,
                lower_bound_scale=lower_bound_scale,
                lower_bound_offset=lower_bound_offset,
                upper_bound_scale=upper_bound_scale,
                upper_bound_offset=upper_bound_offset,
                lower_bound_parent_name=lower_bound_parent_name,
                upper_bound_parent_name=upper_bound_parent_name,
                scale_type="log",
            )
            nodes[sweep_arg.arg] = node
        else:
            raise ValueError(
                f"Invalid sweep arg {sweep_arg}. Must specify exactly one of random_linear or random_log"
            )

        if (
            node.lower_bound_parent_name is None
            and node.upper_bound_parent_name is None
        ):
            roots.append(node)

    # Build the graph by adding edges between nodes.
    # This in O(n^2) but if you have enough nodes to make this 
    # a problem then you probably need to re-evaluate your life choices.
    for parent_node in nodes.values():
        for child_node in nodes.values():
            if parent_node == child_node:
                continue

            if (
                parent_node.name == child_node.lower_bound_parent_name
                or parent_node.name == child_node.upper_bound_parent_name
            ):
                parent_node.add_child(child_node)

    # Check for cycles in the graph
    check_for_cycles(nodes, roots)

    # Check that all nodes are connected.
    for node in roots:
        node.sample_and_propagate()
    print(nodes)
    for node in nodes.values():
        if node.sample is None:
            raise ValueError(
                f"Node {node.name} is not connected to the graph. "
                "Make sure all nodes are connected."
            )

    for node in roots:
        node.clear_tree_samples()

    return nodes, roots


def get_sample_dict(
    rng: np.random.RandomState,
    roots: list[SweepedParamNode],
    nodes: dict[str, SweepedParamNode],
) -> dict[str, float | int]:
    """Get a sample dict from the sweep DAG."""
    for node in roots:
        node.clear_tree_samples()

    for node in roots:
        node.sample_and_propagate()

    sample_dict = {}
    for node in nodes.values():
        sample_dict[node.name] = node.get_sample()

    return sample_dict


def run_sweep_job(sweep_job: SweepJob, script: str, dry_run: bool = False):
    if isinstance(sweep_job.sweep_args, ParameterSweep):
        parameter_sweeps = [sweep_job.sweep_args]
    else:
        parameter_sweeps = sweep_job.sweep_args

    if not parameter_sweeps and sweep_job.budget > 1:
        raise ValueError("No sweep args provided")

    rng = np.random.RandomState(sweep_job.seed)
    command = f"{sys.executable} {script} {sweep_job.fixed_args}"
    results = {}

    hparam_dag, hparam_dag_roots = build_sweep_dag(parameter_sweeps, rng)
    logging.info(
        f"Built sweep DAG with {len(hparam_dag)} nodes and {len(hparam_dag_roots)} roots"
    )
    logging.info(f"Roots: {hparam_dag_roots}")
    logging.info(f"Nodes: {hparam_dag}")

    for i in range(sweep_job.budget):
        sweep_arg_strs = []

        output_dir_params = {}

        sampled_values = get_sample_dict(rng, hparam_dag_roots, hparam_dag)

        for arg_name, arg_value in sampled_values.items():
            sweep_arg_strs.append(f"--{arg_name}={arg_value}")
            output_dir_params[arg_name] = arg_value

        output_dir_suffix = sweep_job.output_dir_suffix.format(**output_dir_params)
        output_dir = os.path.join(sweep_job.output_dir, output_dir_suffix)

        logging.info(f"Hparam Sweep {i + 1}/{sweep_job.budget}")
        logging.info(f"Sampled values: {sweep_arg_strs}")
        sweep_arg_str = " ".join(sweep_arg_strs)
        command_with_sweep_args = (
            f"{command} {sweep_arg_str} --{sweep_job.output_dir_argname}={output_dir}"
        )

        logging.info(f"Running command:\n {command_with_sweep_args}")
        if dry_run:
            logging.info("Dry run. Not running command")
            runtime_formatted = "Dry run"
        else:
            start_time = time.time()
            subprocess.run(command_with_sweep_args.split())
            run_time = time.time() - start_time
            runtime_formatted = format_duration(run_time)

        try:
            results_file_path = os.path.join(output_dir, sweep_job.results_file)
            result = get_result(results_file_path, sweep_job.metric)
        except:
            result = "failed"
        logging.info(
            f"Args:{output_dir_suffix}\nResult: {result}\nRuntime: {runtime_formatted}"
        )

        results[output_dir_suffix] = {
            "result": result,
            "runtime": runtime_formatted,
            "args": sampled_values,
        }
    return results


def annotate_with_best(results):
    """Add a new entry to sweep results showing what the best result is."""
    filtered_results = {
        k: v["result"] for k, v in results.items() if v["result"] != "failed"
    }
    if filtered_results:
        best_key = max(filtered_results, key=filtered_results.get)  # type: ignore
        best_value = results[best_key]["result"]
        best_args = results[best_key]["args"]
    else:
        best_key = None
        best_value = None
        best_args = None

    return {
        "best": {"args": best_args, "value": best_value, "key": best_key},
        "results": results,
    }


def log_sweep_config(sweep_job: SweepJob, script: str):
    """Save the sweep job config to the output dir."""
    payload = {"sweep_job": sweep_job.model_dump(), "script": script}
    with open(os.path.join(sweep_job.output_dir, "sweep_job.json"), "w") as f:
        json.dump(payload, f, indent=4)


if __name__ == "__main__":
    config = tyro.cli(Config)

    with open(config.sweep_file) as f:
        sweep_job = json.load(f)

    sweep_job = SweepJob(**sweep_job)

    # Check that output dir is empty
    os.makedirs(sweep_job.output_dir, exist_ok=True)
    # assert (
    #     len(os.listdir(sweep_job.output_dir)) == 0
    # ), f"Output dir {sweep_job.output_dir} is not empty. Contents: {os.listdir(sweep_job.output_dir)}."

    timestamp = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(sweep_job.output_dir, f"hparam-sweeper-{timestamp}.log")
            ),
        ],
    )
    logging.info(f"Starting sweep exp with {sweep_job.output_dir=}")
    logging.info(f"Config: {config}")
    logging.info(f"SweepJob: {sweep_job}")
    validate_sweep_job(sweep_job)

    log_sweep_config(sweep_job, config.script)
    result = run_sweep_job(sweep_job, config.script, config.dry_run)

    report = annotate_with_best(result)
    logging.info("Report %s", report)

    logging.info(
        f"Writing report to {os.path.join(sweep_job.output_dir, 'report.json')}"
    )
    with open(os.path.join(sweep_job.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=4)

    if sweep_job.best_run_fixed_args:
        best_args = report["best"]["args"]
        logging.info(f"Running with best hparam values:\n {best_args}")

        best_arg_str = " ".join([f"--{k}={v}" for k, v in best_args.items()])
        command = f"{sys.executable} {config.script} {sweep_job.best_run_fixed_args} {best_arg_str}"
        logging.info(f"Running command:\n {command}")
        if not config.dry_run:
            subprocess.run(command.split())
        else:
            logging.warning("Dry run. Not running command")
