import os
import random
import numpy as np
import pandas as pd
import torch
from ete3 import Tree
from tqdm import tqdm

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed_everything] Seed={seed} applied.")

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_full_image_path_from_roi_path(roi_path: str, roi_folder_name: str, full_image_folder_name: str) -> str:
    try:
        roi_filename = os.path.basename(roi_path)
        image_id_str = roi_filename.split('_')[0]
        full_image_filename = f"{image_id_str}.png"

        path_parts = roi_path.split(os.sep)
        try:
            roi_folder_index = path_parts.index(roi_folder_name)
            path_parts[roi_folder_index] = full_image_folder_name
            full_path = os.path.join(os.sep.join(path_parts[:-1]), full_image_filename)
        except ValueError:
            base_dir = os.path.dirname(os.path.dirname(roi_path))
            full_path = os.path.join(base_dir, full_image_folder_name, full_image_filename)

        return full_path
    except Exception as e:
        print(f"CRITICAL Error in get_full_image_path_from_roi_path for ROI '{roi_path}': {e}")
        raise

def get_distance_matrix(label_encoder, tree_path, matrix_path, max_dist):
    if os.path.exists(matrix_path):
        print(f"Loading pre-computed distance matrix from {matrix_path}")
        df = pd.read_csv(matrix_path, index_col=0)
        ordered_labels = label_encoder.classes_
        return torch.tensor(df.loc[ordered_labels, ordered_labels].values, dtype=torch.float32)

    print(f"Distance matrix not found at {matrix_path}. Creating it now...")
    tree = Tree(tree_path, format=1)
    label2node = {node.name: node for node in tree.traverse("preorder") if node.name}
    
    num_classes = len(label_encoder.classes_)
    dist_matrix_np = np.full((num_classes, num_classes), max_dist, dtype=np.float32)

    for i, class_a in enumerate(tqdm(label_encoder.classes_, desc="Building Distance Matrix")):
        for j, class_b in enumerate(label_encoder.classes_[i:], start=i):
            node_a = label2node.get(class_a)
            node_b = label2node.get(class_b)
            if node_a and node_b:
                dist = tree.get_distance(node_a, node_b, topology_only=True)
                dist = min(int(round(dist)), max_dist)
            else:
                dist = max_dist
            dist_matrix_np[i, j] = dist_matrix_np[j, i] = dist
    
    df = pd.DataFrame(dist_matrix_np, index=label_encoder.classes_, columns=label_encoder.classes_)
    df.to_csv(matrix_path)
    print(f"Distance matrix saved to {matrix_path}")
    
    return torch.tensor(dist_matrix_np, dtype=torch.float32)
