import enum


class Filenames(enum.StrEnum):
    GEN_CONFIG = "gen_config.json"
    TRAIN_CONFIG = "train_config.yaml"
    STAR_CONFIG = "star_config.json"
    COMMAND = "command.sh"

    SOLUTIONS = "solution_attempts.json"
    CORRECT_GENERATIONS = "correct_generations.json"
    AGGREGATE = "aggregate_data.jsonl"
    RESULTS = "results.json"

    PROMPTS = "prompts.json"
    REFERENCE_ANSWER = "ref_answers.json"
    REFERENCE_SOLUTIONS = "ref_solutions.json"


class GenJsonKeys(enum.StrEnum):
    """Record keys in the generation json file."""

    PROBLEM = "problem"
    REFERENCE_ANSWER = "reference_answer"
    ATTEMPTS = "attempts"
    IS_CORRECT = "is_correct"
    PROMPT = "prompt"


class TrainingJsonKeys(enum.StrEnum):
    """Record keys in the training jsonl file."""

    PROMPT = "prompt"
    RESPONSE = "response"
