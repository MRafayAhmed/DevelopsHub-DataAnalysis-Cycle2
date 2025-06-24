import kagglehub

# Download latest version
path = kagglehub.dataset_download("shubham3112/cnn-daily-mail-dataset")

print("Path to dataset files:", path)