import tensorflow_datasets as tfds
import torch
import numpy as np

# Load the dataset using TensorFlow Datasets
dataset, info = tfds.load('smallnorb', data_dir='data', download=False, split='train', with_info=True)

# Extract images and labels
images = []
labels = []
for example in dataset:
    images.append(example['image'].numpy())
    labels.append(example['label_category'].numpy())

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Convert to PyTorch tensors
images_tensor = torch.tensor(images, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Print the shape of the tensors
print("Images shape:", images_tensor.shape)
print("Labels shape:", labels_tensor.shape)
