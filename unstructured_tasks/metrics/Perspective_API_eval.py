"""Evaluates comment incoherence and other attributes using the Perspective API."""

from googleapiclient import discovery
import os
import json
import re
import time
import json
from datetime import datetime
from tqdm import tqdm
import dataclasses
import tyro


@dataclasses.dataclass
class Config:
    base_directory: str = "comments_ablations/equal_temp"
    output_directory: str = "ablation_coherence/"
    api_key: str = "<your_api_key>"
    articles_per_temp: int = 100
    comments_per_article: int = 15
    discovery_service_url: str = (
        "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
    )

    def __post_init__(self):
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)


def extract_key_from_dirname(dirname):
    """
    Extract key from directory name based on the pattern.
    For directories with 'ft_X_Y', returns tuple (X, Y)
    For directories with 'ft_X', returns tuple (X,)
    For other directories, returns the directory name
    """
    # Check if it's a directory with ft_X_Y pattern
    match = re.search(r"ft_(\d+\.?\d*)_(\d+\.?\d*)", dirname)
    if match:
        return (float(match.group(1)), float(match.group(2)))

    # Check if it's a directory with ft_X pattern
    match = re.search(r"ft_(\d+\.?\d*)", dirname)
    if match:
        return (float(match.group(1)),)

    # If no numbers found, return the full directory name
    return dirname


def load_all_comments(base_dir):
    """
    Load comments.json from each subdirectory and organize them by directory pattern

    Args:
        base_dir: Base directory containing the output directories

    Returns:
        dict: Keys are either tuples of floats or strings, values are the contents
              of the corresponding comments.json files
    """
    comments_data = {}

    # List all output directories
    for dirname in os.listdir(base_dir):
        # Construct path to comments.json
        comments_path = os.path.join(base_dir, dirname, "comments.json")

        # Skip if not a directory or if comments.json doesn't exist
        if not os.path.isdir(os.path.join(base_dir, dirname)) or not os.path.exists(
            comments_path
        ):
            continue

        # Extract key from directory name
        key = extract_key_from_dirname(dirname)

        # Load and store the comments data
        try:
            with open(comments_path, "r") as f:
                comments_data[key] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading {comments_path}: {e}")
            continue

    return comments_data


def analyze_comments_per_article(
    comments,
    tuple_key,
    client,
    max_comments=None,
    output_file=False,
    request_counter=None,
):
    """
    Analyze comments using Perspective API with quota management.

    Args:
        comments (list): List of comments to analyze
        tuple_key (tuple): The tuple key these comments belong to
        client: Perspective API client
        max_comments (int, optional): Maximum number of comments to process
        output_file (str, optional): Path to save the JSON output
        request_counter (dict, optional): Dictionary of API requests across calls

    Returns:
        list: Processed comments with their scores
    """

    # Initialize or use provided request counter
    if request_counter is None:
        request_counter = {"count": 0, "last_reset": time.time()}

    def check_quota_and_wait():
        """Check if we're near quota limit and wait if necessary"""
        current_time = time.time()
        # Reset counter if a minute has passed since last reset
        if current_time - request_counter["last_reset"] >= 60:
            request_counter["count"] = 0
            request_counter["last_reset"] = current_time

        # If we're at the quota limit, wait for the next minute
        if request_counter["count"] >= 60:
            wait_time = 60 - (current_time - request_counter["last_reset"])
            wait_time *= 1.5
            if wait_time > 0:
                print(
                    f"\nQuota reached. Waiting {wait_time:.1f} seconds for next window"
                )
                time.sleep(wait_time)
            request_counter["count"] = 0
            request_counter["last_reset"] = time.time()

    # Limit comments if max_comments is specified
    if max_comments is not None:
        comments = comments[:max_comments]

    # Create default output filename if none provided
    if output_file:
        tuple_str = "_".join(str(x) for x in tuple_key)
        output_file = f"perspective_results_{tuple_str}.json"

    processed_comments = []
    failed = 0

    print(f"Processing {len(comments)} comments for tuple {tuple_key}")

    for i, comment in enumerate(comments, 1):
        try:
            # Check quota before making request
            check_quota_and_wait()

            # Create API request
            analyze_request = {
                "comment": {"text": str(comment)},
                "languages": ["en"],
                "requestedAttributes": {
                    "TOXICITY": {},
                    "SEVERE_TOXICITY": {},
                    "IDENTITY_ATTACK": {},
                    "INSULT": {},
                    "THREAT": {},
                    "PROFANITY": {},
                    "INCOHERENT": {},
                    "ATTACK_ON_AUTHOR": {},
                    "UNSUBSTANTIAL": {},
                    "ATTACK_ON_COMMENTER": {},
                    "INFLAMMATORY": {},
                    "LIKELY_TO_REJECT": {},
                    "OBSCENE": {},
                    "SPAM": {},
                },
            }

            # Make API call
            response = client.comments().analyze(body=analyze_request).execute()
            request_counter["count"] += 1

            # Extract scores
            scores = {
                attr: response["attributeScores"][attr]["summaryScore"]["value"]
                for attr in analyze_request["requestedAttributes"]
                if attr in response.get("attributeScores", {})
            }

            # Store result
            processed_comments.append({"text": comment, "perspective_scores": scores})

            print(
                f"Processed comment {i}/{len(comments)} (API calls this minute: {request_counter['count']})"
            )

            # Small delay between requests
            time.sleep(0.1)

        except Exception as e:
            print(f"Failed to analyze comment {i}: {str(e)}")
            failed += 1
            continue

    # Prepare output dictionary with metadata
    output_data = {
        "tuple_key": list(tuple_key),  # Convert tuple to list for JSON serialization
        "tuple_key_str": "_".join(str(x) for x in tuple_key),
        "timestamp": datetime.now().isoformat(),
        "total_comments": len(comments),
        "processed_comments": len(processed_comments),
        "failed_comments": failed,
        "results": processed_comments,
    }

    # Save to JSON file with indentation
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"\nCompleted!")
        print(f"Successfully processed: {len(processed_comments)}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {output_file}")

    return processed_comments


def main(cfg: Config):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=cfg.api_key,
        discoveryServiceUrl=cfg.discovery_service_url,
        static_discovery=False,
    )

    all_comments = load_all_comments(cfg.base_directory)
    scores = {}
    request_counter = {"count": 0, "last_reset": time.time()}

    # Add tqdm wrapper around the main loop
    for key in tqdm(list(all_comments.keys()), desc="Processing temperature points"):
        articles_scores = {}
        if len(key) == 1:
            all_article_comments = all_comments[key]["direct"]["comments"]
        else:
            all_article_comments = all_comments[key]["keywords"]["comments"]

        # Add nested tqdm for article processing
        for i in tqdm(
            range(len(all_article_comments[0 : cfg.articles_per_temp])),
            desc=f"Processing articles for temp {key}",
            leave=False,
        ):
            comments = all_article_comments[i]
            processed = analyze_comments_per_article(
                comments,
                key,
                client,
                max_comments=cfg.comments_per_article,
                request_counter=request_counter,
            )
            articles_scores[i] = processed
        scores[str(key)] = articles_scores

        output_file = (
            f"{cfg.output_directory}/API_scores_{cfg.articles_per_temp}_key_{key}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        scores = {}

    config = {
        "articles_per_temp": cfg.articles_per_temp,
        "comments_per_article": cfg.comments_per_article,
        "output_directory": cfg.output_directory,
        "base_directory": cfg.base_directory,
    }
    with open(
        os.path.join(cfg.output_directory, "config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
