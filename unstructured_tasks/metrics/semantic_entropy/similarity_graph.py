"""Makes a similarity graph from a similarity matrix and saves the comment clusters."""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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


def create_similarity_graph(similarity_matrix, comments):
    G = nx.Graph()

    # Add nodes
    for i, comment in enumerate(comments):
        G.add_node(i, comment=comment)

    # Add edges for reciprocal similarities
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(n):  # We only need to check upper triangle
            if similarity_matrix[i, j] == 1 and similarity_matrix[j, i] == 1:
                G.add_edge(i, j)

    return G


def visualize_graph(G, output_file):
    plt.figure(figsize=(20, 16))

    # Identify connected and isolated nodes
    connected_nodes = set(node for node in G.nodes() if G.degree(node) > 0)
    isolated_nodes = set(G.nodes()) - connected_nodes

    # Use a force-directed layout with adjusted parameters
    pos = nx.spring_layout(G, k=0.9, iterations=90)

    # Adjust positions of isolated nodes to bring them closer
    center_x = sum(x for x, y in pos.values()) / len(pos)
    center_y = sum(y for x, y in pos.values()) / len(pos)

    for node in isolated_nodes:
        x, y = pos[node]
        pos[node] = (0.9 * (x - center_x) + center_x, 0.9 * (y - center_y) + center_y)

    # Compute node sizes based on degree (connectivity)
    node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]

    # Draw the graph with thicker black edges
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=1.0,  # Full opacity
        width=2.0,  # Thicker edges
        edge_color="black",
    )  # Black edges

    # Draw nodes with black borders
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="lightblue",
        alpha=0.8,
        linewidths=2.0,  # Thicker node borders
        edgecolors="black",
    )  # Black node borders

    # Add labels with only the index
    labels = {node: str(node) for node in G.nodes()}

    # Draw labels with black text and bold font
    nx.draw_networkx_labels(
        G, pos, labels, font_size=12, font_weight="bold", font_color="black"
    )

    plt.title("Comment Similarity Graph", fontsize=20, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph visualization saved to: {output_file}")


def save_clustered_comments(G, comments, output_file="clustered_comments.txt"):
    # Get connected components (clusters)
    connected_components = list(nx.connected_components(G))

    # Get isolated nodes
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 2]

    with open(output_file, "w", encoding="utf-8") as f:
        cluster_num = 1

        # Write multi-node clusters first
        for component in connected_components:
            if len(component) > 1:  # Multi-node clusters
                f.write(f"Cluster {cluster_num}:\n")
                for node in component:
                    f.write(f"  {node}: {comments[node]}\n")
                f.write("\n---\n\n")
                cluster_num += 1

        # Write single-node clusters (isolated nodes)
        for node in isolated_nodes:
            f.write(f"Cluster {cluster_num}:\n")
            f.write(f"  {node}: {comments[node]}\n")
            f.write("\n---\n\n")
            cluster_num += 1


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comments_file = os.path.join(current_dir, "prompts", "comments.txt")
    similarity_matrix_file = os.path.join(
        current_dir, "comment_similarity_matrix_t0.1.npy"
    )

    # Load comments and similarity matrix
    comments = load_comments(comments_file)[:15]
    similarity_matrix = np.load(similarity_matrix_file)

    # Create graph
    G = create_similarity_graph(similarity_matrix, comments)

    # Print graph information
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # Visualize graph
    output_file = os.path.join(current_dir, "similarity_graph_indices.png")
    visualize_graph(G, output_file)

    # Save clustered comments to file
    clustered_comments_file = os.path.join(current_dir, "clustered_comments.txt")
    save_clustered_comments(G, comments, clustered_comments_file)
    print(f"\nClustered comments saved to: {clustered_comments_file}")

    # Print connected components
    connected_components = list(nx.connected_components(G))
    print(f"\nNumber of connected components: {len(connected_components)}")

    for i, component in enumerate(connected_components, 1):
        print(f"\nComponent {i}:")
        for node in component:
            print(f"  - Node {node}: {comments[node]}...")
    # Print edges of mutual similarity
    print("\nEdges of mutual similarity:")
    for edge in G.edges():
        i, j = edge
        print(f"Node {i} and Node {j} are mutually similar:")
        print(f"  Node {i}: {comments[i]}")
        print(f"  Node {j}: {comments[j]}")
        print()
