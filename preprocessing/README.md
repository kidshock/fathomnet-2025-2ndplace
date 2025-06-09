# Preprocessing Scripts

This directory contains scripts used for data preparation before the main training pipeline is run.

## `create_taxonomy_artifacts.py`

This script generates two crucial files from the competition's metadata: a taxonomic tree and a pairwise distance matrix. These artifacts are essential for understanding the hierarchical relationships between classes and can be used to implement custom loss functions or for advanced validation.

### What it Does

1.  **Fetches Hierarchies**: It reads the 79 class names from `data/dataset_train.json`. For each class, it queries the WoRMS database (via the `fathomnet` library) to get its full taxonomic lineage (species, genus, family, etc.).
2.  **Builds Tree**: It assembles these lineages into a single comprehensive tree structure using the `ete3` library.
3.  **Calculates Distances**: It post-processes the tree to align with the competition's scoring metric, which only considers 7 specific ranks.
4.  **Saves Artifacts**:
    *   `taxonomy_tree.nh`: The full tree saved in [Newick format](https://en.wikipedia.org/wiki/Newick_format).
    *   `distance_matrix.csv`: A CSV file where each cell `(i, j)` contains the taxonomic distance between class `i` and class `j`.

### How to Run

Ensure you have installed the necessary dependencies from the root `requirements.txt` file. Then, from the root directory of the repository, run the following command:

```bash
python preprocessing/create_taxonomy_artifacts.py
```

The script will use the default file paths (`data/dataset_train.json`) and save the outputs to the `data/` directory. You can specify different paths using command-line arguments: