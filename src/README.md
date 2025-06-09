##  How to Run Training

### . Prepare Data
Place all data (CSVs + image folders) into the data/ folder.

Ensure your train.csv has a column called full_image_path.

Run this to create the taxonomy tree and distance matrix:

 - python preprocessing/create_taxonomy_artifacts.py


### Train the Model

 - python -m src.train
 
Note: The -m flag is important – it lets Python properly load the files inside src/.