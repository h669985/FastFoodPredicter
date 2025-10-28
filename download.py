import kagglehub

# Download latest version
path = kagglehub.dataset_download("tan5577/nutritonal-fast-food-dataset")

print("Path to dataset files:", path)