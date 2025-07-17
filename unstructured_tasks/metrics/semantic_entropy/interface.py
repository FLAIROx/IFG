"""Utilities interface for calculating semantic entropy from comments."""

from typing import Optional
from .semantic_comments import classify_comment_pairs
from . import semantic_comments
from . import similarity_graph
import os
import dotenv
import networkx as nx
import numpy as np
from numpy import typing as npt
import vllm


def semantic_entropy_from_graph(graph: nx.Graph):
    """Get the connected components of the graph."""
    connected_components = nx.connected_components(graph)
    distribution = [len(component) for component in connected_components]
    distribution = np.array(distribution)
    distribution = distribution / distribution.sum()
    EPS = 1e-9
    print("entropy distribution", distribution)
    print("log", np.log(distribution + EPS))
    entropy = -np.sum(distribution * np.log(distribution + EPS))
    return entropy


def semantic_entropy_from_sim_matrix(matrix: npt.NDArray):
    """Get the semantic entropy from a similarity matrix."""

    comments = [""] * matrix.shape[0]
    graph = similarity_graph.create_similarity_graph(matrix, comments)
    similarity_graph.visualize_graph(graph, "graph.png")
    return semantic_entropy_from_graph(graph)


def semantic_entropy_from_comments(
    comments: list[str], semantic_entropy_prompt: str, llm: Optional[vllm.LLM] = None
):
    """Get the semantic entropy from a list of comments."""
    sim_matrix = classify_comment_pairs(comments, semantic_entropy_prompt, llm=llm)
    return semantic_entropy_from_sim_matrix(sim_matrix)


if __name__ == "__main__":
    dotenv.load_dotenv()
    import numpy as np

    comments = [
        "I love potatoes",
        "I think potatoes are the best",
        "Kittens are cute",
    ]
    file_dir = os.path.dirname(os.path.abspath(__file__))
    semantic_entropy_prompt_path = os.path.join(
        file_dir, "prompts", "semantic_clusters.txt"
    )

    semantic_entropy_prompt = semantic_comments.read_prompt_file(
        semantic_entropy_prompt_path
    )

    print(semantic_entropy_from_comments(comments, semantic_entropy_prompt))

    # prompt = semantic_comments.read_prompt_file(semantic_clusters_prompt_path)

    # def get_entropy_from_matrix(matrix):
    #     graph = similarity_graph.create_similarity_graph(matrix, comments)
    #     return semantic_entropy_from_graph(graph)

    # similarity_matrix = np.eye(3)
    # entropy_eye = get_entropy_from_matrix(similarity_matrix)
    # similarity_matrix = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # entropy_bimodal = get_entropy_from_matrix(similarity_matrix)
    # similarity_matrix = np.ones((3, 3))
    # entropy_single_cluster = get_entropy_from_matrix(similarity_matrix)

    # print(f"Entropy for eye matrix: {entropy_eye}")
    # print(f"Entropy for bimodal matrix: {entropy_bimodal}")
    # print(f"Entropy for single cluster matrix: {entropy_single_cluster}")
