"""Compute semantic entropy for generated stories and intents."""

from __future__ import annotations

import os
# Disable vLLM usage stats to avoid disk quota issues
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["VLLM_USAGE_STATS"] = "0"

import dataclasses
import json
from typing import Any, Optional

import numpy as np
from tqdm import tqdm
import tyro
import vllm
from vllm import SamplingParams
import networkx as nx

from unstructured_tasks.metrics.semantic_entropy import similarity_graph

EPS = 1e-9


@dataclasses.dataclass
class Config:
    input_dir: str = "stories_generation/outputs/qwen3-32b"
    intent_prompt_path: str = (
        "unstructured_tasks/metrics/semantic_entropy/prompts/semantic_clusters_stories_intent.txt"
    )
    story_prompt_path: str = (
        "unstructured_tasks/metrics/semantic_entropy/prompts/semantic_clusters_stories_content.txt"
    )
    intent_matching_prompt_path: str = (
        "unstructured_tasks/metrics/semantic_entropy/prompts/semantic_clusters_stories_intent_matching.txt"
    )
    output_path: str = "stories_generation/outputs/semantic_entropy_gemma_12b.json"
    similarity_model: str = "google/gemma-3-12b-it"
    similarity_temperature: float = 0.1
    similarity_batch_size: int = 100
    max_files: Optional[int] = None


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_story_payloads(input_dir: str, max_files: Optional[int]) -> list[dict[str, Any]]:
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    files = [
        file_name
        for file_name in os.listdir(input_dir)
        if file_name.endswith(".json") and file_name != "config.json"
    ]
    files.sort()
    if max_files is not None:
        files = files[:max_files]

    payloads: list[dict[str, Any]] = []
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))
    return payloads


def _clean_outline_text(text: str) -> str:
    """Clean outline text by removing markdown artifacts."""
    return text.strip()


def _clean_story_text(text: str) -> str:
    """Clean story text by removing markdown artifacts."""
    return text.strip()


def _similarity_matrix_with_reasoning(
    texts: list[str],
    prompt: str,
    llm: vllm.LLM,
    temperature: float,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Compute similarity matrix and return reasoning for each pair.

    Returns:
        Tuple of (similarity_matrix, reasoning_dict) where reasoning_dict
        maps "i,j" -> {"response": str, "similar": bool}
    """
    if len(texts) <= 1:
        return np.eye(len(texts)), {}

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=512,
        stop=["###"],
    )

    num_texts = len(texts)
    similarity_matrix = np.zeros((num_texts, num_texts))
    reasoning_dict: dict[str, dict[str, Any]] = {}

    prompts_batch = []
    indices = []

    for i in range(num_texts):
        for j in range(num_texts):
            classification_prompt = prompt.format(
                comment1=texts[i], comment2=texts[j]
            )
            prompts_batch.append(classification_prompt)
            indices.append((i, j))

            if len(prompts_batch) == batch_size:
                outputs = llm.generate(prompts_batch, sampling_params)
                for k, output in enumerate(outputs):
                    response = output.outputs[0].text.strip()
                    idx_i, idx_j = indices[k]
                    is_similar = "response: similar" in response.lower()
                    if is_similar:
                        similarity_matrix[idx_i, idx_j] = 1
                    reasoning_dict[f"{idx_i},{idx_j}"] = {
                        "response": response,
                        "similar": is_similar,
                    }
                prompts_batch = []
                indices = []

    # Process remaining
    if prompts_batch:
        outputs = llm.generate(prompts_batch, sampling_params)
        for k, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            idx_i, idx_j = indices[k]
            is_similar = "response: similar" in response.lower()
            if is_similar:
                similarity_matrix[idx_i, idx_j] = 1
            reasoning_dict[f"{idx_i},{idx_j}"] = {
                "response": response,
                "similar": is_similar,
            }

    return similarity_matrix, reasoning_dict


def _compute_intent_story_matching_matrix_with_reasoning(
    intents: list[str],
    stories: list[str],
    prompt_template: str,
    llm: vllm.LLM,
    temperature: float,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Compute matching matrix between intents and stories with reasoning.

    Args:
        intents: List of intent texts
        stories: List of story texts
        prompt_template: Template with {intent} and {story} placeholders
        llm: vLLM instance
        temperature: Sampling temperature

    Returns:
        Tuple of (match_matrix, reasoning_dict) where:
        - match_matrix: N x N matrix where element [i,j] is 1 if intent[i] matches story[j]
        - reasoning_dict: maps "i,j" -> {"response": str, "matches": bool}
    """
    n_intents = len(intents)
    n_stories = len(stories)
    reasoning_dict: dict[str, dict[str, Any]] = {}

    if n_intents == 0 or n_stories == 0:
        return np.array([]).reshape(n_intents, n_stories), reasoning_dict

    # Create all prompts for the cross-product
    prompts = []
    for intent in intents:
        for story in stories:
            prompts.append(prompt_template.format(intent=intent, story=story))

    # Generate responses
    tokenizer = llm.get_tokenizer()
    chat_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        chat_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
    )
    results = llm.generate(chat_prompts, sampling_params=sampling_params)

    # Parse responses to get match/mismatch
    match_matrix = np.zeros((n_intents, n_stories))
    for idx, result in enumerate(results):
        response = result.outputs[0].text.strip()
        i = idx // n_stories
        j = idx % n_stories
        # Check for "matches" but not "mismatches"
        response_lower = response.lower()
        is_match = "matches" in response_lower and "mismatches" not in response_lower
        if is_match:
            match_matrix[i, j] = 1
        reasoning_dict[f"{i},{j}"] = {
            "response": response,
            "matches": is_match,
        }

    return match_matrix, reasoning_dict


def _cluster_labels(similarity_matrix: np.ndarray) -> list[int]:
    if similarity_matrix.size == 0:
        return []
    comments = [""] * similarity_matrix.shape[0]
    graph = similarity_graph.create_similarity_graph(similarity_matrix, comments)
    components = list(nx.connected_components(graph))
    labels = [-1] * similarity_matrix.shape[0]
    for idx, component in enumerate(components):
        for node in component:
            labels[node] = idx
    return labels


def _entropy_from_labels(labels: list[int]) -> float:
    if not labels:
        return 0.0
    counts = np.bincount(np.array(labels, dtype=int))
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + EPS)))


def _conditional_entropy_responses_given_intents(
    matching_matrix: np.ndarray,
    intent_labels: list[int],
    response_labels: list[int],
) -> float:
    """Compute H(response_cluster | intent_cluster) using matching matrix.

    For each intent cluster, find which stories match those intents,
    then compute entropy of the story cluster distribution.

    Args:
        matching_matrix: N x N matrix where [i,j] = 1 if intent i matches story j
        intent_labels: Cluster labels for each intent
        response_labels: Cluster labels for each response

    Returns:
        Conditional entropy H(response_cluster | intent_cluster)
    """
    # if matching_matrix.size == 0 or not intent_labels or not response_labels:
    #     return 0.0

    # intent_labels_arr = np.array(intent_labels, dtype=int)
    # response_labels_arr = np.array(response_labels, dtype=int)
    # n_intents = len(intent_labels_arr)
    # total_entropy = 0.0

    # for cluster_id in np.unique(intent_labels_arr):
    #     # Get indices of intents in this cluster
    #     intent_indices = np.where(intent_labels_arr == cluster_id)[0]
    #     cluster_size = len(intent_indices)

    #     # Find all stories that match any intent in this cluster
    #     matched_response_labels = []
    #     for i in intent_indices:
    #         matched_stories = np.where(matching_matrix[i, :] == 1)[0]
    #         matched_response_labels.extend(response_labels_arr[matched_stories].tolist())

    #     # Compute entropy of response cluster distribution
    #     if matched_response_labels:
    #         cluster_entropy = _entropy_from_labels(matched_response_labels)
    #     else:
    #         cluster_entropy = 0.0

    #     total_entropy += (cluster_size / n_intents) * cluster_entropy
    
    # breakpoint()
    dists = matching_matrix / matching_matrix.sum(axis=1, keepdims=True)
    entropy = -np.sum(dists * np.log(dists + EPS), axis=1)
    total_entropy = entropy.mean()

    return float(total_entropy)


def _conditional_entropy_intents_given_responses(
    matching_matrix: np.ndarray,
    intent_labels: list[int],
    response_labels: list[int],
) -> float:
    """Compute H(intent_cluster | response_cluster) using matching matrix.

    For each response cluster, find which intents match those stories,
    then compute entropy of the intent cluster distribution.

    Args:
        matching_matrix: N x N matrix where [i,j] = 1 if intent i matches story j
        intent_labels: Cluster labels for each intent
        response_labels: Cluster labels for each response

    Returns:
        Conditional entropy H(intent_cluster | response_cluster)
    """
    return _conditional_entropy_responses_given_intents(matching_matrix.T, response_labels, intent_labels)
    # if matching_matrix.size == 0 or not intent_labels or not response_labels:
    #     return 0.0

    # intent_labels_arr = np.array(intent_labels, dtype=int)
    # response_labels_arr = np.array(response_labels, dtype=int)
    # n_responses = len(response_labels_arr)
    # total_entropy = 0.0

    # for cluster_id in np.unique(response_labels_arr):
    #     # Get indices of responses in this cluster
    #     response_indices = np.where(response_labels_arr == cluster_id)[0]
    #     cluster_size = len(response_indices)

    #     # Find all intents that match any story in this cluster
    #     matched_intent_labels = []
    #     for j in response_indices:
    #         matched_intents = np.where(matching_matrix[:, j] == 1)[0]
    #         matched_intent_labels.extend(intent_labels_arr[matched_intents].tolist())

    #     # Compute entropy of intent cluster distribution
    #     if matched_intent_labels:
    #         cluster_entropy = _entropy_from_labels(matched_intent_labels)
    #     else:
    #         cluster_entropy = 0.0

    #     total_entropy += (cluster_size / n_responses) * cluster_entropy

    # return float(total_entropy)


def _group_by_style_description(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group stories by their outline style description."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in payload.get("stories", []):
        outline_info = entry.get("outline", {})
        description = outline_info.get("style_description", "unknown")
        grouped.setdefault(description, []).append(entry)
    return grouped


def main(cfg: Config) -> None:
    intent_prompt = _load_prompt(cfg.intent_prompt_path)
    story_prompt = _load_prompt(cfg.story_prompt_path)
    intent_matching_prompt = _load_prompt(cfg.intent_matching_prompt_path)

    llm = vllm.LLM(model=cfg.similarity_model)

    payloads = _load_story_payloads(cfg.input_dir, cfg.max_files)
    results: dict[str, Any] = {
        "input_dir": cfg.input_dir,
        "intent_prompt_path": cfg.intent_prompt_path,
        "story_prompt_path": cfg.story_prompt_path,
        "intent_matching_prompt_path": cfg.intent_matching_prompt_path,
        "metrics_by_prompt": {},
        "averaged_metrics_by_style": {},  # Will hold averaged metrics across all stories
    }

    # Collect metrics per style across all stories for averaging
    style_metrics_collector: dict[str, list[dict[str, float]]] = {}

    for payload in tqdm(payloads, desc="Prompts", unit="prompt"):
        prompt_id = payload.get("prompt_id", "unknown")
        prompt_text = payload.get("prompt", "")
        grouped = _group_by_style_description(payload)

        per_style_metrics: dict[str, Any] = {}
        for style_description, entries in tqdm(
            grouped.items(),
            desc=f"{prompt_id} styles",
            unit="style",
            total=len(grouped),
            leave=False,
        ):
            # Extract outline texts (intents) and story texts (responses)
            outline_texts = [
                _clean_outline_text(entry.get("outline", {}).get("text", ""))
                for entry in entries
            ]
            response_texts = [
                _clean_story_text(entry.get("story", ""))
                for entry in entries
            ]

            # Compute outline/intent similarity using intent prompt
            intent_sim, intent_similarity_reasoning = _similarity_matrix_with_reasoning(
                outline_texts,
                intent_prompt,
                llm,
                cfg.similarity_temperature,
                cfg.similarity_batch_size,
            )

            # Compute story similarity using story content prompt
            response_sim, response_similarity_reasoning = _similarity_matrix_with_reasoning(
                response_texts,
                story_prompt,
                llm,
                cfg.similarity_temperature,
                cfg.similarity_batch_size,
            )

            # Compute intent-story matching matrix
            matching_matrix, matching_reasoning = _compute_intent_story_matching_matrix_with_reasoning(
                outline_texts,
                response_texts,
                intent_matching_prompt,
                llm,
                cfg.similarity_temperature,
            )

            intent_labels = _cluster_labels(intent_sim)
            response_labels = _cluster_labels(response_sim)

            # Diagonal match rate: how often does story[i] match outline[i]
            diagonal_matches = np.diag(matching_matrix) if matching_matrix.size > 0 else np.array([])
            match_rate = float(diagonal_matches.mean()) if len(diagonal_matches) > 0 else 0.0

            metrics = {
                "count": len(entries),
                "entropy_intents": _entropy_from_labels(intent_labels),
                "entropy_responses": _entropy_from_labels(response_labels),
                "conditional_entropy_responses_given_intents": _conditional_entropy_responses_given_intents(
                    matching_matrix, intent_labels, response_labels
                ),
                "conditional_entropy_intents_given_responses": _conditional_entropy_intents_given_responses(
                    matching_matrix, intent_labels, response_labels
                ),
                "intent_story_match_rate": match_rate,
                # Save the texts and reasoning
                "outline_texts": outline_texts,
                "response_texts": response_texts,
                "intent_labels": intent_labels,
                "response_labels": response_labels,
                "intent_similarity_reasoning": intent_similarity_reasoning,
                "response_similarity_reasoning": response_similarity_reasoning,
                "intent_story_matching_reasoning": matching_reasoning,
            }

            per_style_metrics[style_description] = metrics

            # Collect for averaging across stories (only numeric metrics)
            if style_description not in style_metrics_collector:
                style_metrics_collector[style_description] = []
            style_metrics_collector[style_description].append({
                "entropy_intents": metrics["entropy_intents"],
                "entropy_responses": metrics["entropy_responses"],
                "conditional_entropy_responses_given_intents": metrics["conditional_entropy_responses_given_intents"],
                "conditional_entropy_intents_given_responses": metrics["conditional_entropy_intents_given_responses"],
                "intent_story_match_rate": metrics["intent_story_match_rate"],
            })

        results["metrics_by_prompt"][prompt_id] = {
            "prompt": prompt_text,
            "style_description_metrics": per_style_metrics,
        }

    # Compute averaged metrics by style (across all stories)
    for style_description, metrics_list in style_metrics_collector.items():
        n_stories = len(metrics_list)
        averaged = {
            "n_stories": n_stories,
            "entropy_intents": sum(m["entropy_intents"] for m in metrics_list) / n_stories,
            "entropy_responses": sum(m["entropy_responses"] for m in metrics_list) / n_stories,
            "conditional_entropy_responses_given_intents": sum(
                m["conditional_entropy_responses_given_intents"] for m in metrics_list
            ) / n_stories,
            "conditional_entropy_intents_given_responses": sum(
                m["conditional_entropy_intents_given_responses"] for m in metrics_list
            ) / n_stories,
            "intent_story_match_rate": sum(m["intent_story_match_rate"] for m in metrics_list) / n_stories,
        }
        results["averaged_metrics_by_style"][style_description] = averaged

    output_dir = os.path.dirname(cfg.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main(tyro.cli(Config))
