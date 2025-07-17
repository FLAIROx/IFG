"""Classifies comments into semantic clusters using a prompt."""

import logging
import os
import numpy as np
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt


def read_prompt_file(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()


def load_comments(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    comments = []
    current_comment = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # If we find an Author line
        if "Author:" in line:
            # If we have collected a comment, add it to our comments list
            if current_comment:
                comments.append(" ".join(current_comment))
                current_comment = []

            # Move to next line after Author
            i += 1

            # Keep reading until we hit another Author or end of file
            while i < len(lines) and "Author:" not in lines[i]:
                line = lines[i].strip()
                if line:  # Only add non-empty lines
                    current_comment.append(line)
                i += 1
            continue

        i += 1

    # Don't forget to add the last comment if there is one
    # Add the final comment to the list if it exists
    if current_comment:
        comments.append(" ".join(current_comment))

    return comments


def classify_comment_pairs(
    comments,
    prompt,
    temperature=0.1,
    batch_size=100,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    llm=None,
):
    if llm is None:
        llm = LLM(model=model_name)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=512,  # Increased to allow for reasoning
        stop=["###"],
    )

    num_comments = len(comments)
    similarity_matrix = np.zeros((num_comments, num_comments))

    # Open a file to save the responses
    with open("responses.txt", "w", encoding="utf-8") as response_file:
        prompts = []
        indices = []

        for i in range(num_comments):
            for j in range(num_comments):
                classification_prompt = prompt.format(
                    comment1=comments[i], comment2=comments[j]
                )
                prompts.append(classification_prompt)
                indices.append((i, j))

                # Process in batches
                if len(prompts) == batch_size:
                    outputs = llm.generate(prompts, sampling_params)
                    for k, output in enumerate(outputs):
                        response = output.outputs[0].text.strip()
                        i, j = indices[k]

                        # Split by ### and get the content after the 10th occurrence
                        parts = prompts[k].split("###")
                        if len(parts) > 10:
                            temp = "###" + parts[-1]  # Keep the last relevant part
                        else:
                            temp = response  # Keep full response if not enough ###

                        # Save the filtered response to the file
                        response_file.write(
                            f"Prompt:\n{temp}\n\nResponse:\n{response}\n\n---\n\n"
                        )

                        # Look for the final "Response: " line to determine similarity
                        if "response: similar" in response.lower():
                            similarity_matrix[i, j] = 1

                        # Print full response including reasoning (optional)
                        logging.debug(f"\nComparing comments {i} and {j}:")
                        logging.debug(response)
                        # logging.debug("---")

                    # Clear the batch
                    prompts = []
                    indices = []

        # Process any remaining prompts
        if prompts:
            outputs = llm.generate(prompts, sampling_params)
            for k, output in enumerate(outputs):
                response = output.outputs[0].text.strip()
                i, j = indices[k]

                # Split by ### and get the content after the 10th occurrence
                parts = prompts[k].split("###")
                if len(parts) > 10:
                    temp = "###" + parts[-1]  # Keep the last relevant part
                else:
                    temp = response  # Keep full response if not enough ###

                # Save the filtered response to the file
                response_file.write(
                    f"Prompt:\n{temp}\n\nResponse:\n{response}\n\n---\n\n"
                )

                # Look for the final "Response: " line to determine similarity
                if "response: similar" in response.lower():
                    similarity_matrix[i, j] = 1

                # Print full response including reasoning (optional)
                logging.debug(f"\nComparing comments {i} and {j}:")
                logging.debug(response)
                # logging.debug("---")

    return similarity_matrix


def visualize_similarity_matrix(similarity_matrix, comments, temperature):
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Similarity")
    plt.title("Comment Similarity Matrix")
    plt.xlabel("Comment Index")
    plt.ylabel("Comment Index")

    # Add comment texts as tick labels (truncated if too long)
    max_label_length = 50
    truncated_comments = [
        c[:max_label_length] + "..." if len(c) > max_label_length else c
        for c in comments
    ]
    plt.xticks(range(len(comments)), truncated_comments, rotation=90, ha="right")
    plt.yticks(range(len(comments)), truncated_comments)

    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(
        current_dir, f"similarity_matrix_visualization_t{temperature}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSimilarity matrix visualization saved to: {output_file}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comments_file = os.path.join(current_dir, "prompts", "comments.txt")
    semantic_clusters_prompt_path = os.path.join(
        current_dir, "prompts", "semantic_clusters.txt"
    )

    # Check if the token is set in the environment
    huggingface_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not huggingface_token:
        print("Warning: HUGGING_FACE_HUB_TOKEN is not set in the environment.")
    else:
        print("HUGGING_FACE_HUB_TOKEN is set in the environment.")

    # Load existing comments
    comments = load_comments(comments_file)
    print(f"Loaded {len(comments)} comments.")
    comments = comments[15:30]
    # for c in comments:
    #     print(c)
    #     print("==================================================================END OF COMMENT")

    temperature = 0.1

    # Classify comment pairs
    semantic_clusters_prompt = read_prompt_file(semantic_clusters_prompt_path)
    print(semantic_clusters_prompt)
    similarity_matrix = classify_comment_pairs(
        comments, semantic_clusters_prompt, temperature
    )

    # Save similarity matrix to a file
    similarity_matrix_file = os.path.join(
        current_dir, f"comment_similarity_matrix_t{temperature}.npy"
    )
    np.save(similarity_matrix_file, similarity_matrix)
    print(f"\nSimilarity matrix saved to: {similarity_matrix_file}")

    # Visualize the similarity matrix
    visualize_similarity_matrix(similarity_matrix, comments, temperature)

    # Optional: Print a sample of the similarity matrix
    print("\nSample of the similarity matrix (first 5x5):")
    print(similarity_matrix[:5, :5])
