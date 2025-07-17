import os
from typing import Optional
import sys
import typing

import cattrs
import tyro
import yaml
import glob


ConfigClass = typing.Type
Config = typing.Any

def tyro_cli_with_yaml_support(config_cls: ConfigClass):
    """Extend tyro CLI with YAML configuration support.

    This function allows loading default values from a YAML file when using tyro CLI.
    It looks for a '--yaml' argument in the command line arguments, loads the YAML file,
    and uses it as default values for the tyro CLI configuration. Arguments provided
    in the command line will override the YAML values.
    Args:
        config_cls (ConfigClass): A dataclass that defines the configuration structure.
            Must be compatible with both tyro and cattrs.
    Returns:
        The instantiated configuration object of type config_cls, with values from either
        the YAML file (if provided) and/or command line arguments.
    Example:
        ```python
        @dataclass
        class MyConfig:
            param1: str
            param2: int
        config = tyro_cli_with_yaml_support(MyConfig)
        ```
        Can be called from command line as:
        ```bash
        python script.py --yaml config.yaml
        # or
        python script.py --yaml=config.yaml
        ```
    """
    # I promise I will to learn to use hydra for my next project.

    # Find the yaml config path argument if it exists.
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--yaml":
            yaml_path = sys.argv[i + 1]
            # Remove the --yaml and the path from sys.argv
            sys.argv.pop(i)
            sys.argv.pop(i)
            break
        elif sys.argv[i].startswith("--yaml="):
            yaml_path = sys.argv[i].removeprefix("--yaml=")
            # Remove the --yaml=path from sys.argv
            sys.argv.pop(i)
            break
    else:
        yaml_path = None

    # Remove the yaml config arg from sys.argv
    if yaml_path is not None:
        # Load the yaml config
        with open(yaml_path, "r") as f:
            yaml_str = f.read()
        defaults = yaml.safe_load(yaml_str)

        # We used cattrs to convert the nested dictionaries to the dataclass.
        converter = cattrs.Converter(forbid_extra_keys=True)
        default = converter.structure(defaults, config_cls)  # Convert to dataclass.
    else:
        default = None

    config = tyro.cli(config_cls, default=default)
    return config

def _expand_glob_patterns(filenames_or_globs: list[str], base_dir: str) -> set[str]:
    """Replace glob patterns with matching filenames."""
    allowed_files = set()
    for pattern in filenames_or_globs:
        # Check if the pattern is a glob by looking for wildcard characters
        if any(c in pattern for c in ['*', '?', '[']):
            # Expand the glob pattern to get matching files
            matching_files = [os.path.basename(f) for f in glob.glob(os.path.join(base_dir, pattern))]
            allowed_files.update(matching_files)
        else:
            # If it's not a glob pattern, add it directly
            allowed_files.add(pattern)
    return allowed_files


def prepare_output_dir(output_dir: str, pre_existing_files: Optional[list[str]] = None):
    """Create output dir if it does not exist.

    We assert that the directory path is not a file and that
    the directory is empty (aside from files excluded by pre_existing_files).

    Args:
        output_dir (str): Path to the output directory.
        pre_existing_files (list[str]): List of files that are allowed to exist in the output directory.
            List entries can either be filenames or glob patterns.
    """
    if pre_existing_files is None:
        pre_existing_files = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return

    assert not os.path.isfile(
        output_dir
    ), f"Provided output directory path {output_dir} is a file."

    # Check if the directory is empty (with the exception of pre_existing_files).
    # Expand glob patterns and create a set of all allowed files

    # Replace the pre_existing_files list with the expanded set
    allowed_files = _expand_glob_patterns(pre_existing_files, output_dir)
    offending_files = [f for f in os.listdir(output_dir) if f not in allowed_files]

    assert not offending_files, (
        f"Provided output directory <{output_dir}> is not empty."
        f"First offending file: <{offending_files and offending_files[0]}>. Allowed exclusions: <{pre_existing_files}>"
    )


def flatten_config(config: Config | dict, prefix: Optional[str] = None) -> dict:
    """Flatten a nested configuration dataclass to a flat dictionary.

    Args:
        config (ConfigClass): The configuration dataclass.
        prefix (str): The prefix to add to the keys of the
            flattened dictionary.
    Returns:
        dict: The flattened configuration dictionary.
    """
    flat_config = {}
    unstructured = cattrs.unstructure(config)
    for key, value in unstructured.items():
        if isinstance(value, dict):
            flat_config.update(
                flattened=flatten_config(value, f"{prefix}.{key}" if prefix else key)
            )
        else:
            flat_config[f"{prefix}.{key}" if prefix else key] = value
    return flat_config
