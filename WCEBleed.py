

# !wget "https://zenodo.org/record/7548320/files/WCEBleedGen.zip"
# !unzip "/content/WCEBleedGen.zip"

# Import necessary libraries
import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter,ImageEnhance,ImageOps, ImageDraw
from skimage.util import random_noise
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Subset
import cv2

"""### Data Augmentation.py

"""

# Define a custom data augmentation class called WCEImageTransforms
class WCEImageTransforms:
    def __init__(self, rotation_degrees, blur_parameters):
        """
        Initialize the data augmentation class with rotation and blur parameters.

        Args:
            rotation_degrees (list): A list of degrees for rotation augmentation.
            blur_parameters (list): A list of tuples containing blur parameters (w, sigma).
        """
        self.rotation_degrees = rotation_degrees
        self.blur_parameters = blur_parameters

    def __call__(self, img):
        """
        Apply random rotation and Gaussian blur to an input image.

        Args:
            img (PIL.Image): Input PIL image to be augmented.

        Returns:
            img_tensor (torch.Tensor): Transformed image as a PyTorch tensor.
        """
        # Randomly decide whether to rotate the image
        should_rotate = random.choice([True, False])
        if should_rotate:
            # Randomly select a degree from the set and rotate the image
            random_degree = random.choice(self.rotation_degrees)
            img = img.rotate(random_degree, expand=True)

        # Randomly decide whether to apply Gaussian blur
        should_blur = random.choice([True, False])
        if should_blur:
            # Randomly select blur parameters (w, sigma) from the set
            random_w, random_sigma = random.choice(self.blur_parameters)

            # Apply Gaussian blur to the image
            img = img.filter(ImageFilter.GaussianBlur(radius=random_sigma))

        # Convert the PIL image to a PyTorch tensor with resizing and normalization
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.4605, 0.2799, 0.1762],
                                                        std=[0.2111, 0.1551, 0.1120])])
        img_tensor = trans(img)

        return img_tensor


"""### Dataset.py

"""

# Define a custom dataset class called WCEClassDataset
class WCEClassDataset(Dataset):
    def __init__(self, root_dir, num_models=1):
        """
        Initialize the custom dataset.

        Args:
            root_dir (str): Root directory containing subfolders for different classes.
            num_models (int): Number of models (1 by default) used for multi-input data.
        """
        super().__init__()
        self.root_dir = root_dir
        self.num_models = num_models

        # List of subfolders (class names)
        self.classes = [os.path.join(root_dir, 'bleeding', 'Images'), os.path.join(root_dir, 'non-bleeding', 'images')]

        # Initialize lists to hold image paths and labels
        self.image_paths = []
        self.labels = []

        # Load image paths and labels
        for class_idx, class_dir in enumerate(self.classes):
            image_files = os.listdir(class_dir)
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                self.image_paths.append(image_path)
                self.labels.append(class_idx)  # bleeding images given label 0 and non-bleeding 1

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a single data sample and its label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (PIL.Image): The image as a PIL image.
            label (int): The corresponding label (0 for bleeding, 1 for non-bleeding).
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)

        return image, label

# Define a custom subset dataset class called WCEClassSubsetDataset
class WCEClassSubsetDataset(Dataset):
    def __init__(self, original_dataset, subset_indices, transform=None):
        """
        Initialize a subset dataset based on an original dataset.

        Args:
            original_dataset (WCEClassDataset): The original dataset.
            subset_indices (list): List of indices specifying the subset of data to use.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.original_dataset = original_dataset
        self.subset_indices = subset_indices
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the subset."""
        return len(self.subset_indices)

    def __getitem__(self, idx):
        """
        Get a single data sample and its label from the subset.

        Args:
            idx (int): Index of the sample to retrieve from the subset.

        Returns:
            image (PIL.Image or list of PIL.Image): The image(s) as PIL image(s).
            label (int): The corresponding label.
        """
        # Get an item from the subset
        image, label = self.original_dataset[self.subset_indices[idx]]

        # Apply the transform if it is provided
        if self.transform:
            images = []
            for i in range(self.original_dataset.num_models):
                images.append(self.transform(image))

            return images, label

        return image, label


# Import necessary libraries
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy

# Define the training function that calculates precision, recall, and F1 score
def train(model, train_loader, val_loader, optimizer, lr_scheduler, criterion, device, num_epochs, save_dir, model_name):
    # Initialize variables to keep track of the best validation performance
    best_val_acc = 0.0
    best_epoch = 0
    best_precision = 0.0
    best_recall = 0.0
    best_f1_score = 0.0

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0.0
        total_train = 0.0

        # Create a progress bar for training
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} (Train)')
        for inputs, labels in train_bar:
            # Prepare labels and move data to the specified device (e.g., GPU)
            labels = labels.to(torch.float).unsqueeze(1).to(device)

            # Zero the gradients, perform forward and backward pass, and update weights
            optimizer.zero_grad()
            outputs = model(inputs[0].to(device))
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            # Update training loss and compute accuracy
            train_loss += loss.item()

            # Define a threshold (e.g., 0.5) for binary classification
            threshold = 0.5

            # Apply thresholding to get binary predictions
            binary_predictions = [(prob > threshold).type(torch.int) for prob in outputs]
            binary_predictions = (torch.tensor(binary_predictions)).unsqueeze(1).to(device)

            # Update total and correct counts for accuracy calculation
            total_train += labels.size(0)
            correct_train += (binary_predictions == labels).sum().item()

            # Update the progress bar
            train_bar.set_postfix(loss=train_loss / (train_bar.n + 1), accuracy=correct_train / total_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_predictions = []  # To store predictions for precision and recall calculation
        val_targets = []      # To store actual targets for precision and recall calculation

        # Create a progress bar for validation
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)')
        with torch.no_grad():
            for inputs, labels in val_bar:
                # Prepare labels and move data to the specified device (e.g., GPU)
                labels = labels.to(torch.float).unsqueeze(1).to(device)

                # Forward pass and compute validation loss
                outputs = model(inputs[0].to(device))
                loss = criterion(outputs, labels).to(device)
                val_loss += loss.item()

                # Define a threshold (e.g., 0.5) for binary classification
                threshold = 0.5

                # Apply thresholding to get binary predictions
                binary_predictions = [(prob > threshold).type(torch.int) for prob in outputs]
                binary_predictions = (torch.tensor(binary_predictions)).unsqueeze(1).to(device)

                # Extend lists with predictions and actual targets for evaluation
                val_predictions.extend(binary_predictions.cpu().numpy().astype(numpy.float))
                val_targets.extend(labels.cpu().numpy())

                # Update total and correct counts for accuracy calculation
                total_val += labels.size(0)
                correct_val += (binary_predictions == labels).sum().item()

                # Update the progress bar
                val_bar.set_postfix(loss=val_loss / (val_bar.n + 1), accuracy=correct_val / total_val)
        
        # Calculate precision, recall, and F1 score for the current epoch
        precision = precision_score(val_targets, val_predictions)
        recall = recall_score(val_targets, val_predictions)
        f1 = f1_score(val_targets, val_predictions)

        # Adjust the learning rate based on the learning rate scheduler
        lr_scheduler.step()

        # Save the model checkpoint if it has the best validation accuracy
        if correct_val > best_val_acc:
            best_precision = precision
            best_recall = recall
            best_f1_score = f1
            best_val_acc = correct_val
            best_epoch = epoch

            # Save the model checkpoint to the specified save directory
            checkpoint_path = os.path.join(save_dir, f'{model_name}_best2.pth')
            torch.save(model.state_dict(), checkpoint_path)

    # Print relevant statistics and evaluation metrics after training
    print(len(val_predictions))
    print(len(val_targets))
    print(f'Best model found at epoch {best_epoch + 1} with validation accuracy of {best_val_acc / total_val * 100:.2f}%')
    print(f'Best Precision: {best_precision:.2f}, Recall: {best_recall:.2f}, F1 Score: {best_f1_score:.2f}')


"""### Model.py

"""

# Import necessary libraries
import torch.nn as nn
import torchvision

# Define a custom neural network model based on VGG16 architecture
class CustomVgg16(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomVgg16, self).__init__()

        # Load the pre-trained VGG16 model from torchvision
        self.mobilenet_v2 = torchvision.models.vgg16(pretrained=False)  # Pre-trained weights not used here

        # Modify the classifier (fully connected) layer
        in_features = self.mobilenet_v2.classifier[6].in_features
        self.mobilenet_v2.classifier[6] = nn.Linear(in_features, num_classes)  # Replace the classifier's last layer

        # Define a sigmoid activation layer for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        # Apply the modified VGG16 model and sigmoid activation
        return self.sigmoid(self.mobilenet_v2(x))

    def initialize_weights(self):
        """
        Initialize the weights of the custom model's classifier layers.

        This function uses Xavier initialization for linear layers' weights.
        """
        for layer in self.mobilenet_v2.classifier[1].modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.xavier_uniform_(layer.bias)

"""### Main.py

"""

# Define the main function
def main():
    # Specify the root directory where the dataset is located
    root_dir = '../datasets/WCEBleedGen'
    
    # Check if CUDA (GPU) is available, otherwise use CPU
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the dataset
    dataset = WCEClassDataset(root_dir=root_dir)
    
    # Define training parameters
    num_epochs = 100
    batch_size = 8
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42
    save_dir = './'
    model_name = 'WCE_class'

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    # Shuffle dataset indices if needed
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    # Define data transformations for training and validation
    rotation_degrees = [90, 180]
    blur_parameters = [(5, 9)]
    train_transform = WCEImageTransforms(rotation_degrees, blur_parameters)
    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4605, 0.2799, 0.1762],
                                                               std=[0.2111, 0.1551, 0.1120])])

    # Create training and validation datasets
    train_dataset = WCEClassSubsetDataset(dataset, train_indices, train_transform)
    valid_dataset = WCEClassSubsetDataset(dataset, val_indices, valid_transform)
    
    # Print the length of the training dataset
    print(f"len_dataset:{len(train_dataset)}")

    # Create data loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # Initialize the custom VGG16 model and move it to the specified device (CPU/GPU)
    custom_vgg16 = CustomVgg16(num_classes=1).to(device)

    # Define optimizer settings
    lr = 0.001
    weight_decay = 1.0e-4

    # Create the SGD optimizer with momentum and weight decay
    optimizer = torch.optim.SGD(custom_vgg16.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Define the loss criterion for binary classification (BCELoss)
    import torch.nn as nn
    criterion = nn.BCELoss()

    # Create a learning rate scheduler with a step decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Start the training process
    train(custom_vgg16, train_loader, validation_loader, optimizer, lr_scheduler, criterion, device, num_epochs, save_dir, model_name)

# Execute the main function when the script is run

# if __name__ == '__main__':
#     main()

"""### Test.py

"""

# Check if CUDA (GPU) is available, otherwise use CPU

# Define a function to perform testing
def test(model, test_dir):
    # Set the model to evaluation mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    # Define the directory to save predicted images
    output_directory = './test_data_1'
    os.makedirs(output_directory, exist_ok=True)
    
    # Define a transformation to preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the model's input size
        transforms.ToTensor(),
    ])

    # List all files in the test directory
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    ground_truth = [0] * len(image_files)  # Ground truth labels (assuming all are initially labeled as 0)
    threshold = 0.5  # Threshold for binary classification
    probabilities = []
    results = []
    test_acc = 0.0
    
    for image_file in image_files:
        # Load and preprocess the image
        image_path = os.path.join(test_dir, image_file)
        image = Image.open(image_path)
        image_tensor = preprocess(image).unsqueeze(0)  # Add a batch dimension
        image_tensor = image_tensor.to(device)

        # Perform a forward pass through the model
        with torch.no_grad():
            outputs = model(image_tensor)
            pred_labels = int((outputs.item() > threshold))
            label_text = "Bleeding" if pred_labels == 0 else "Non-Bleeding"
        
        # Display the image with the label and output tensor
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Label: {label_text}\nOutput Tensor: {outputs.item()}", fontsize=14)
        plt.axis("off")
        plt.savefig(os.path.join(output_directory, image_file))
        
        # Store the predicted labels
        probabilities.append(pred_labels)
        results.append((image_file, pred_labels))
    
    # Calculate test accuracy
    probabilities = torch.tensor(probabilities).to(device)
    ground_truth = torch.tensor(ground_truth).to(device)
    test_acc = (probabilities == ground_truth).sum().item()
    test_acc = test_acc / len(image_files)
    print(test_acc)

    # Save predictions to a CSV file
    csv_filename = "predictions.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image ID", "Label"])
        csv_writer.writerows(results)

# Call the main function
if __name__ == '__main__':
    main()
# main()

# Define the path to the trained model
    model_path = "./vgg_without_weight_100_epochs_93.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model
    model = CustomVgg16()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Define the directory containing test images
    test_directory = '../datasets/Auto-WCEBleedGen Challenge Test Dataset/Test Dataset 1'

    # Call the test function to make predictions
    test(model, test_directory)
