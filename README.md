# ISIC 2024 - Skin Cancer Detection with 3D-TBP
This Repository Contains My different approaches for Kaggle competetion [ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge/overview).

## Task
To develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). The image quality resembles close-up smartphone photos, which are regularly submitted for telehealth purposes. The binary classification algorithm developed here is designed to support settings lacking specialized dermatological care, with the goal of improving triage and facilitating early detection of skin cancer.

## Dataset
The training dataset consists of **401,059** images of skin lesions, accompanied by a table providing additional metadata for each image. The dataset is highly imbalanced, with 400,666 negative examples and only **393 positive examples**. For more details about the dataset and the competition, please visit the [competition page](https://www.kaggle.com/competitions/isic-2024-challenge/data?select=train-metadata.csv).

## Solutions
This section contains different solutions for the competetion. 
### Solution 1 - Light Gradient Boosting Machine (LGBM)
The model was trained using Light Gradient Boosting Machine (LGBM) on the tabular data provided alongside the images. This approach achieved a performance score of **0.126**. The detailed implementation and results can be found in the notebook [lightgbm.ipynb](https://github.com/Ashaz4994/ISIC_2024/blob/main/lightgbm.ipynb).

### Solution 2- CatBoost+Feature Eingineering
A CatBoost model was trained using features engineered from the provided tabular data. The engineered features include various combinations of the original columns to enhance model performance. It achieved a score of **0.144**. The full implementation and details are available in the notebook [catboost-feature-eingineering.ipynb](https://github.com/Ashaz4994/ISIC_2024/blob/main/catboost-feature-eingineering.ipynb).

### Solution 3- CNN
Multiple pretrained models were evaluated by training a classifier on top of their feature representations. Architectures such as ResNet and EfficientNet were among those tested. The results for the different models are summarized in the table below. Training procedures are documented in [training-cnn.ipynb](https://github.com/Ashaz4994/ISIC_2024/blob/main/training-cnn.ipynb), and inference steps are provided in [cnn-inference.ipynb](https://github.com/Ashaz4994/ISIC_2024/blob/main/cnn-inference.ipynb). I also tried Upsampling, Downsampling and Augmentations.
| Model         | Best Score  |
|---------------|--------|
| ResNet18      | 0.137  |
| DINOv2        | 0.138  |
| EfficientNet  | 0.134  |
| DenseNet      | 0.118  |

### Solution 4- Hybrid Architecture (Image + Metadata Fusion)
This model combines image and tabular metadata inputs through a dual-branch architecture:

- **Image Branch:** Processed via a CNN (e.g., ResNet, EfficientNet) to extract visual features.
- **Metadata Branch:** Tabular data (e.g. lesion attributes) is fed into a feedforward neural network.

The outputs of both branches are concatenated and passed to a final classifier.  
This fusion approach achieved a score of **0.156**, demonstrating improved performance over single-modality models.

The inference code is in [inference_notebook.ipynb](https://github.com/Ashaz4994/ISIC_2024/blob/main/inference_notebook.ipynb) and the Training code is in [met_img_train.ipynb](https://github.com/Ashaz4994/ISIC_2024/blob/main/met_img_train.ipynb)
