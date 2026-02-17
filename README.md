# DeepShield
Explainable AI Framework for Deepfake Detection and Media Authenticity.

## Key Features

- 🔍 **Deepfake Image Detection**
  - Binary classification: Real vs Fake
  - CNN-based model using EfficientNet backbone

- 📊 **Authenticity Score**
  - Outputs a probability score (0–1) indicating how likely an image is real
  - Avoids rigid binary decisions

- 🧠 **Explainability (Grad-CAM)**
  - Visual heatmaps highlighting regions that influenced the model’s decision
  - Improves transparency and trust

- 📈 **Comprehensive Evaluation**
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC
  - Confusion Matrix

## 🗂️ Project Structure

DeepShield/
│
├── datasets/
│ └── images/
│       ├── fake/
│       └── real/
│
├── models/
│ ├── image_model.py
│ └── image_model.pth
│
├── utils/
│ ├── preprocess.py
│ ├── inference.py
| ├── multicrop.py
│ └── explainability.py
│
├── outputs/
│ └── gradcam_result.jpg
│
├── test_ datasets/
│ └── images/
│
├── train_image.py
├── test_image.py
├── requirements.txt
└── README.md

## Install Dependencies:
```
pip install -r requirements.txt
```

## Training the Model: 
```
python train_image.py
```

Saves the trained model to => models/image_model.pth

## Image Inference + Explainability
```
python infer_image.py
```

Grad-CAM heatmap saved to => outputs/gradcam_result.jpg



