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
# Best Model performed in validation set (Classification Model link)
https://drive.google.com/file/d/1D_femI9n9LmLY2ettvcSdoONiYO83yg_/view?usp=sharing
# Best Model performed in validation set (Detection Model link)
https://drive.google.com/file/d/1E7Oh-rI7G3wdUB50T6c-hB-kdiSGvJku/view?usp=sharing
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
![img- (1214)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/a26917fe-34f0-43f0-a564-7929ffedb557)
![img- (1152)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/7550acdb-6003-4952-be95-1373f96593a0)
![img- (1139)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/3abe9cd2-eae1-410c-acc2-08d5bd92aea5)
![img- (1122)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/7ecb06cc-2b33-4503-8b2b-53f884ddbd71)
![img- (939)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/bdfd14fd-2ce6-405b-b92b-cbc1f5116eb0)
![img- (154)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/d4bfbbd7-7161-4930-9d6f-8cfb9b537239)
![img- (122)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/08892c82-d951-45a1-8fd1-73f53357b9f6)
![img- (84)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/d35e6277-a09a-4769-8483-fc2085b931ed)
![img- (42)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/34c65983-5cf2-43bf-af37-518d7a543400)
![img- (21)](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/9cbafe75-cf40-45c3-bdee-b2efd67b0c0b)



# Screenshots/ pictures of achieved interpretability plot of any 10 best images selected from validation dataset.
![img- (42)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/5ea756f0-ba1e-4d82-893d-7029e5c16196)
![img- (41)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/2928d4d3-b5e8-4248-b260-45b1f37ee98f)
![img- (40)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/f70baeca-6b52-4821-95b8-60f97ef5930b)
![img- (39)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/33176fa0-862c-4fe5-b435-606859cbec83)
![img- (38)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/88779242-68d6-44f6-b608-a863fd9b33ee)
![img- (27)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/c0185e54-fc4c-48ec-a5f5-85b31460f697)
![img- (26)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/28143773-cfe1-4894-aad0-9369b8dfd0be)
![img- (24)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/7f7c5c8c-46a2-487c-9050-05556a341840)
![img- (19)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/7cea83f7-064e-46c5-9d18-41f016a535e4)
![img- (18)_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/1dd7ce18-6e72-4fae-8dab-bfb5e703084a)


# Screenshots/pictures of any 5 best images selected from testing dataset 1 and 2 separately showing its classification and detection (bounding box with confidence level).
## Test Dataset 1
![A0046](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/142d9841-7d10-4931-908c-6dd98cd1b08c)
![A0037](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/b1358d15-0fe3-46c4-99e9-68e14ce253e2)
![A0027](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/d68cee5f-d718-4ab6-bd0e-fee9f71e2f10)
![A0020](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/30da807d-dd17-40bb-b7b1-2d62d7787ce4)
![A0002](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/0e233dd7-ced1-4ac2-9f4c-87a2f05c5e5c)


## Test Dataset 2
![A0392](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/d950207c-8cad-43ea-a0e3-b7d4d5a51508)
![A0386](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/89ff64cf-ccc1-41d1-878e-4002eb409f62)
![A0292](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/2e4ba650-966c-4a2c-b446-25f1df39c64e)
![A0157](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/524828c9-4950-4b39-a258-b4691e73bc6a)
![A0136](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/e59459bf-aa39-49f3-9c24-71c75ac12727)



# Screenshots/ pictures of achieved interpretability plot of any 5 best images selected from testing dataset 1 and 2 separately.
## Test Dataset 1
![A0026_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/73dda009-ba3a-488e-be3a-63c8015350e5)
![A0021_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/ec3e986d-8e6c-40cc-afc7-6b287a9953ff)
![A0009_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/2ed45aab-f326-4c89-b3e5-f54bd4441fcb)
![A0007_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/7bc6cd33-7dd6-4ffc-a05d-279caf7a0c8a)
![A0001_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/7922a787-a095-4d1b-bfeb-4b715888aa18)


## Test Dataset 2

![A0448_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/2b606d5e-4aee-45b7-af23-5e031830cf36)
![A0362_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/2cad1479-7d39-4691-9f46-d8b653b07359)
![A0290_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/2e4adf86-6101-472a-ac03-4c5c5ecb03d6)
![A0269_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/63c8b1d0-5a1d-4e27-b188-5e0eb70e2d84)
![A0195_explanation](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/f89e854e-0ecc-4f89-bd51-9f75877c4842)


# Result graph for detection (Epochs VS various metrics)

![results](https://github.com/bikashrpadhy/Vinayakas/assets/61779823/adbf8739-39f0-4b63-9da7-7bdb576f7585)




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
