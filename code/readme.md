This is code base solves the problem "Mitosis Detection in Breast Cancer HistoPathology Images"

The structure of the directory is below. Each folder is an experiment and a different approach to solve this complex machine learing challenge.

The challenge is, given an image of a breast cancer biopsy histopathology slide, can you detect which cells are mitotic? I extended this challenge to also detect cells that are non-mitotic to reduce false positives. I have tried to solve this challenge as a binary and multiclass problem.

binary
    - Standalone code to solve this problem to label each image as Mitosis and Non-Mitosis using CNN

blue_threshold
    - Standalone code that preprocesses each image by applying a blue threshold to reduce stain variation across the dataset
    - Multiclass problem of labelling each image as mitosis, non-mitosis, background using CNN
    
clustering 
    - Standalone code that divides the dataset into 3 clusters using KMeans based on the stain variation to increase labelling accuracy

image_analysis.py 
    - Scans each image and breaks it down to the given image size
    - Steps through the image using the given xstep and ystep sises
    - Extends boundries by mirroring over the edge to ensure scanning window always results in the desired image size
    - Labels each image based on true mitosis/non-mitosis labels provided in the dataset

multiclass
    - Standalone code that processes each image and using a CNN to predict if each image has a mitotic cell, non-mitotic cell or neither in it. Input image sizes are 100x100px.

testing
    - Unpickle a model
    - Create test set
    - Predict on the test set
    - Calculate metrics

alt_code
    - Variations of the code that are worth preserving for future investigation

image_search_engine
    - Image search engine to find similar images
