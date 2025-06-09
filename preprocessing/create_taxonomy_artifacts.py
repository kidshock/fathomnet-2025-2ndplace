
# These generated artifacts can be used during training, for example, to implement a custom
# hierarchical loss function or for analysis.
# IMPORTANT : This approach is heavily inspired from this notebook: https://www.kaggle.com/code/pake-taxonomic-distance-matrix by https://www.kaggle.com/platypusbear (Great thanks for sharing!)
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from ete3 import Tree
from fathomnet.api import worms

ACCEPTED_RANKS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def get_class_names(json_path: str) -> list:
    print(f"Loading class names from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    classes = [x['name'] for x in data['categories']]
    print(f"Found {len(classes)} classes.")
    return classes

def recursive_child_snatcher(anc):
    children = [x.name for x in anc.children]
    childrens_ranks = [x.rank for x in anc.children]
    if len(anc.children) > 0:
        childrens_children, childrens_childrens_ranks = recursive_child_snatcher(anc.children[0])
        return children + childrens_children, childrens_ranks + childrens_childrens_ranks
    else:
        return children, childrens_ranks

def build_taxonomic_tree(classes: list) -> Tree:
    print("Building taxonomic tree from class list...")
    tree = Tree(name="root")
    tree.rank = "root"
    already_added = {"root"}

    for label in tqdm(classes, desc="Fetching Taxonomies"):
        if label in already_added:
            continue
        try:
            anc = worms.get_ancestors(label)
            path_nodes = [anc.name] + [child for child, rank in zip(*recursive_child_snatcher(anc))]
            path_ranks = [anc.rank] + [rank for child, rank in zip(*recursive_child_snatcher(anc))]
        except Exception as e:
            print(f"Could not fetch ancestors for '{label}': {e}")
            continue
        current_node = tree
        for i in range(len(path_nodes)):
            node_name = path_nodes[i]
            node_rank = path_ranks[i]
            if node_name in already_added:
                current_node = tree.search_nodes(name=node_name)[0]
                continue
            child_node = Tree(name=node_name)
            child_node.rank = node_rank
            current_node.add_child(child_node)
            already_added.add(node_name)
            current_node = child_node
    return tree

def postprocess_tree(tree: Tree, classes: list):
    print("Post-processing tree distances...")
    for node in tree.traverse():
        if node.rank and node.rank.lower() not in ACCEPTED_RANKS:
            node.dist = 0
        if node.name in classes and not node.is_leaf():
            print(f"Warning: Class node '{node.name}' is not a leaf node.")

def create_distance_matrix(tree: Tree, labels: list) -> pd.DataFrame:
    print("Generating pairwise distance matrix...")
    n = len(labels)
    sorted_labels = sorted(labels)
    dist_matrix = np.zeros((n, n))
    label_to_node = {node.name: node for node in tree.traverse()}

    for i, name1 in enumerate(tqdm(sorted_labels, desc="Calculating Distances")):
        for j, name2 in enumerate(sorted_labels):
            if i <= j:
                node1 = label_to_node.get(name1)
                node2 = label_to_node.get(name2)
                if node1 and node2:
                    d = node1.get_distance(node2)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

    df = pd.DataFrame(dist_matrix, index=sorted_labels, columns=sorted_labels)
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate taxonomic artifacts for FathomNet 2025.")
    parser.add_argument(
        "--input_json",
        type=str,
        default="../data/dataset_train.json",
        help="Path to the competition's training JSON file."
    )
    parser.add_argument(
        "--output_tree",
        type=str,
        default="../data/taxonomy_tree.nh",
        help="Path to save the output Newick tree file."
    )
    parser.add_argument(
        "--output_matrix",
        type=str,
        default="../data/distance_matrix.csv",
        help="Path to save the output distance matrix CSV file."
    )
    args = parser.parse_args()

    classes = get_class_names(args.input_json)
    tree = build_taxonomic_tree(classes)
    postprocess_tree(tree, classes)
    tree.write(outfile=args.output_tree, format=1)
    print(f"Taxonomy tree saved to {args.output_tree}")
    dist_df = create_distance_matrix(tree, classes)
    dist_df.to_csv(args.output_matrix)
    print(f"Distance matrix saved to {args.output_matrix}")

if __name__ == "__main__":
    main()
