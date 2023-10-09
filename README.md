# Vinayakas
# Wireless Capsule Endoscopy (WCE) Bleeding Detection

This repository contains code for a bleeding classification and detection system using Wireless Capsule Endoscopy (WCE) images. The system is designed to classify images as either "bleeding" or "non-bleeding." It includes custom data augmentation, dataset management, model training, and testing components. For detection we used YOLOv8 model in which pre-trained `YOLOv8s.pt` was used. The detection model was used with the following hyper-parameters:
`epochs = 100` `conf = 0.25` `iou = 0.4`

## Table of Contents
- [Dataset](#dataset)
- [Getting Started](#gettingstarted)
- [Results](#results)
- [Excel sheet](#excelsheet)
- [Run Using](#runusing)
- [Acknowledgements](#acknowledgements)
- [Methodology_Classification](#Methodology_Classification)
- [Methodology_Detection](#Methodology_Detection)
## Dataset

The dataset used for training and validatating the model can be obtained from the following source:

[Wireless Capsule Endoscopy Bleeding Dataset (WCEBleedGen)](https://zenodo.org/record/7548320)

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone <https://github.com/bikashrpadhy/Vinayakas>
```


# Classification metrics (For validation set)
| Metrics | Values |
| ------------- | ------------- |
|  Accuracy     |    0.9943      |
|  Recall       |    1.00      |
|  Precision    |    0.99      |
|  F1-Score     |    0.99   |


# Detection metrics (For validation set)
| Metrics | Values |
| ------------- | ------------- |
|       Average Precision   |    0.792      |
|  mAP50       | 0.853         |
|  mAP50-95   |  0.641         |
|  IoU   |  0.6948         |

# Screenshots/pictures of any 10 best images selected from validation dataset showing its classification and detection (bounding box with confidence level).
![img- (1214)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/370f6c05-de46-4f9e-9cdf-0fe8f9616e04)
![img- (1152)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/829adf40-785d-4329-bcdf-5d2e302bf6ff)
![img- (1139)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/afe14f92-35a2-4e81-ba51-15d9c7bf53f5)
![img- (1122)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/e5e94148-322d-460f-a608-0c019acc47ec)
![img- (939)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/c32bae3a-0f90-4063-af8d-5cdbddbefaec)
![img- (154)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/779e6ca2-d111-4afa-b734-8179f87452f1)
![img- (122)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/b42213f1-6eba-462e-9c2a-b51fd8568835)
![img- (84)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/68c0f3d6-952d-4916-9e5d-a3b9b945e3f8)
![img- (42)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/549f92dd-328b-4f79-a333-f005b252add9)
![img- (21)](https://github.com/prasadmangala02/Vinayakas/assets/61779823/e9258f9e-639f-4442-8005-44095c6982a9)

# Screenshots/ pictures of achieved interpretability plot of any 10 best images selected from validation dataset.

![img- (42)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/4f9e3fd3-392e-4c3d-b958-2216cc9fdb31)
![img- (41)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/92bc64de-5729-455c-b1c5-4adddba5090b)
![img- (40)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/7af0fb22-7655-445c-b65f-d64dc25a9610)
![img- (39)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/eb738980-e82f-47e7-8fa8-4a19f242a7e7)
![img- (38)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/46fb21f3-d707-490f-ad2e-10903e03f882)
![img- (27)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/d9e16103-d861-4701-ba2a-43bceb4a3a55)
![img- (26)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/826265bc-bc8f-43ef-93a6-0ff1064aff77)
![img- (24)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/7d91cb5b-6755-4861-8415-38209c82fdae)
![img- (19)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/6fa61609-6bf8-413d-89f6-7f1c4cef60f0)
![img- (18)_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/bccd1a41-fe28-4e61-a4f8-18701db72bb0)

# Screenshots/pictures of any 5 best images selected from testing dataset 1 and 2 separately showing its classification and detection (bounding box with confidence level).
## Test Dataset 1
![A0046](https://github.com/prasadmangala02/Vinayakas/assets/61779823/1d23bdfe-0339-4494-a7fd-b9b1ef3a2cd5)
![A0037](https://github.com/prasadmangala02/Vinayakas/assets/61779823/32828422-a633-463d-8987-113fdd0ba7f4)
![A0027](https://github.com/prasadmangala02/Vinayakas/assets/61779823/af35bf1e-ab76-40a3-b5f0-ff215f2b6cbf)
![A0020](https://github.com/prasadmangala02/Vinayakas/assets/61779823/3479bbca-a5ac-4f21-9d20-fd5521ac810e)
![A0002](https://github.com/prasadmangala02/Vinayakas/assets/61779823/cc7adffd-9b5b-471b-880f-f6d1d0485168)


## Test Dataset 2
![A0392](https://github.com/prasadmangala02/Vinayakas/assets/61779823/5c28f4ce-55db-40e2-8de6-47036d2a04b5)
![A0386](https://github.com/prasadmangala02/Vinayakas/assets/61779823/dab96f9f-9f45-4559-a213-5e5fac1d2277)
![A0292](https://github.com/prasadmangala02/Vinayakas/assets/61779823/7b78bc87-2cbe-4c27-a77a-ef20e9b1a3d1)
![A0157](https://github.com/prasadmangala02/Vinayakas/assets/61779823/36924e37-5426-40e3-9bdd-f432b3913b4a)
![A0136](https://github.com/prasadmangala02/Vinayakas/assets/61779823/73897b0d-1002-4930-ac14-fb46626fa647)

# Screenshots/ pictures of achieved interpretability plot of any 5 best images selected from testing dataset 1 and 2 separately.
## Test Dataset 1
![A0026_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/4ddca472-c733-4b10-ad4d-b6d20d4e09b7)
![A0021_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/7aee71db-f827-44fb-ace9-49398bca7daf)
![A0009_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/11d46b4a-d046-44d3-9662-a1bb9a701a59)
![A0007_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/9e7c9f3b-6e8d-418b-ba5a-df2d8f41f157)
![A0001_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/f5458f2b-2d25-42c0-a5fa-6c679088f789)

## Test Dataset 2
![A0448_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/2cfd7573-5cc2-4274-aa75-15743e8a27b9)
![A0362_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/cb55b81f-58ce-4c0b-8c4f-e099685d2954)
![A0290_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/b6d0d6bf-f7f2-4855-8096-1f2360ed7e7c)
![A0269_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/5ef9494d-995b-4c46-8709-4a1da6b81227)
![A0195_explanation](https://github.com/prasadmangala02/Vinayakas/assets/61779823/ebdf3da9-8f16-47aa-bae9-99c81822f051)


# Result graph for detection (Epochs VS various metrics)
![results](https://github.com/prasadmangala02/Vinayakas/assets/61779823/0e602d29-75ba-476f-91d4-c0e2b9068668)


# Excel sheet
Excel sheet contains the image IDs and predicted class labels of testing dataset 1 and 2 `predictions_with_label_test_dataset_1.csv` and `predictions_with_label_test_dataset_2.csv`
# Run Using
`python3 WCEBleed.py`

# Acknowledgement
    The code uses PyTorch for deep learning and torchvision for image transformations.
    The bleeding classification model is based on the VGG16 architecture.
    The bleeding detection model is based on the YOLOv8 architecture.

# Classification Methodology

## 1. Data Acquisition and Preprocessing:
   - The code begins by downloading and unzipping a dataset from a specified URL. The dataset contains images related to bleeding detection in WCE images.

## 2. Data Augmentation:
   - A custom data augmentation class called `WCEImageTransforms` is defined. This class performs `random image rotations` and `Gaussian blur` to augment the dataset. These augmentations are used to generate variations of the original images, which can help improve model generalization.

## 3. Dataset Creation:
   - Two custom dataset classes are defined, `WCEClassDataset` and `WCEClassSubsetDataset`.
   - `WCEClassDataset` loads and organizes the dataset, creating a list of image paths and their corresponding labels (0 for bleeding, 1 for non-bleeding).
   - `WCEClassSubsetDataset` is used to create subsets of the dataset based on specified indices and applies data transformations to the images.

## 4. Training:
   - The code defines a training function that trains a custom VGG16-based neural network model (`CustomVgg16`) on the provided dataset.
   - During training, it uses binary cross-entropy loss (`nn.BCELoss`) as the loss function and Stochastic Gradient Descent (`torch.optim.SGD`) as the optimizer.
   - A learning rate scheduler (`torch.optim.lr_scheduler.StepLR`) is used to adjust the learning rate during training.
   - Training statistics and evaluation metrics (precision, recall, F1 score) are printed during training.
   - The best model based on validation accuracy is saved.

## 5. Model Definition:
   - The custom neural network model `CustomVgg16` is defined based on the VGG16 architecture. The model's classifier (fully connected) layer is modified to output a single value for binary classification.
   - Sigmoid activation is applied to the model's output for binary classification.

## 6. Main Function:
   - The `main()` function orchestrates the entire training process.
   - It loads the dataset, splits it into training and validation sets, defines data transformations, initializes the model, optimizer, and loss criterion, and then starts training.

## 7. Testing:
   - There is a testing function `test` that is defined.
   - This function loads the model and applies it to a directory of test images.
   - It calculates test accuracy and saves the model's predictions and visualizations of the test images.

## 8. Execution:
   - The `main()` function is called when the script is run, triggering the training process.


# Detection Methodology

## 1. Data Download and Extraction:
   - The methodology begins by downloading a zip file from a Zenodo repository using the `wget` command. The zip file contains data related to WCEBleedGen.
   - It then uses the `unzip` command to extract the contents of the downloaded zip file.

## 2. Data Splitting:
   - The code prepares the data for object detection by organizing it into training and validation sets.
   - It defines a root directory containing the data and an output directory for the object detection dataset.
   - Images are shuffled randomly, and a specified percentage (e.g., 20%) of the images are moved to the validation set while the rest are used for training.
   - The data is organized into subdirectories for images and labels in both the training and validation sets.

## 3. Ultralytics Installation:
   - The code installs the Ultralytics library, which is a toolkit for training and deploying object detection models, including YOLO.

## 4. YOLOv8 Training:
   - The code initiates a YOLO training task using Ultralytics with the following parameters:
     - `task`: Specifies that the task is object detection.
     - `mode`: Sets the mode to "train," indicating model training.
     - `model`: Specifies the YOLO model to be used (yolov8s).
     - `data`: Specifies the path to the data configuration file (WCEBleed.yaml) that defines dataset and training settings.
     - `epochs`: Sets the number of training epochs (e.g., 100).
     - `pretrained`: Specifies whether to use pre-trained weights.
     - `conf` and `iou`: Sets confidence and IoU (Intersection over Union) thresholds.

## 5. YOLOv8 Prediction:
   - The code initiates a YOLO prediction task using Ultralytics with the following parameters:
     - `task`: Specifies that the task is object detection.
     - `mode`: Sets the mode to "predict," indicating model prediction.
     - `model`: Specifies the path to the best model performed in the validation set.
     - `source`: Specifies the source directory or file for prediction (test data).
     - `save`: When set to True, it indicates that the predictions will be saved.

## 6. Archiving Training Results:
   - After training and prediction, the code creates a zip archive containing the training results (runs) and saves it as "runs.zip."

## 7. Methodology Summary:
   - The methodology involves data download, dataset preparation, YOLOv8 model training, and object detection on test data using a pre-trained from the command line interface. The code uses the Ultralytics library to facilitate YOLO-based object detection tasks.
