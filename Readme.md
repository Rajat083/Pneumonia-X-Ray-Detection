# 🩺 Pneumonia Detection using Deep Learning

This project implements a Convolutional Neural Network (CNN) to detect Pneumonia from chest X-ray images. The model is trained and evaluated on the publicly available Chest X-Ray Images (Pneumonia) dataset from Kaggle.

# 📌 Project Structure
.
├── Images/                     # Images regarding the Model
├── Models/                     # Saved trained models (H5/Checkpoint files)
├── helper_functions.py         # Utility functions for data loading, preprocessing, and visualization
├── pneumonia_detection.ipynb   # Main Jupyter Notebook (training + evaluation)
└── README.md                   # Project documentation

# 
# 📂 Dataset

The dataset used is the Chest X-Ray Images (Pneumonia). It contains two classes:

Normal: Healthy chest X-rays

Pneumonia: X-rays showing pneumonia infection

The dataset is divided into train, validation, and test sets.

# 🛠️ Features

Data preprocessing: resizing, normalization, augmentation.

CNN model for binary classification (Normal vs Pneumonia).

Evaluation with accuracy, precision, recall, F1-score.

Visualizations: training curves, confusion matrix, prediction confidence.

# 🧠 Model

Architecture: EfficientNetB0 (base model) Custom CNN with Conv2D, MaxPooling, Dropout, Dense layers.

Loss Function: Binary Crossentropy

Optimizer: AdamW

Metrics: Accuracy, Precision, Recall, F1-score

The trained model is saved in the ./Models/ directory.

# 📊 Results & Visualizations

## The following figures illustrate the model performance:

🔹 **Sample X-ray Images**
![alt text](<Images/Screenshot from 2025-08-19 19-43-25.png>)
![alt text](<Images/Screenshot from 2025-08-19 19-43-36.png>)
![alt text](<Images/Screenshot from 2025-08-19 19-43-43.png>)
![alt text](<Images/Screenshot from 2025-08-19 19-43-48.png>)
![alt text](<Images/Screenshot from 2025-08-19 19-43-48.png>)
🔹 **Confusion Matrix**

![alt text](<Images/Screenshot from 2025-08-19 19-41-45.png>)

🔹 **Prediction Confidence Distribution**

![alt text](<Images/Screenshot from 2025-08-19 19-42-16.png>)

🔹 **Training & Validation Accuracy**

![alt text](<Images/Screenshot from 2025-08-19 19-49-26.png>)

🔹 **Training & Validation Loss**

![alt text](<Images/Screenshot from 2025-08-19 19-42-53.png>)

🔹 F1-Scores

![alt text](<Images/Screenshot from 2025-08-19 19-42-02.png>)

# ✅ Conclusion

The CNN model demonstrates strong performance in detecting pneumonia from chest X-rays.

High recall ensures fewer missed pneumonia cases, which is crucial in medical diagnosis.

Future improvements: transfer learning (VGG16, ResNet), explainable AI (Grad-CAM), and larger datasets.

**Accuracy** : 79.5 %  Test accuracy
**Percision** : 0.82
**Recall** : 0.75

**Classification Report**    **precision**    **recall**  **f1-score**   **support**

    BACTERIAL_PNEUMONIA       0.85      0.85      0.85        242

                 NORMAL       0.90      0.73      0.80        234

        VIRAL_PNEUMONIA       0.62      0.81      0.70        148

               accuracy                           0.79        624

              macro avg       0.79      0.79      0.78        624

           weighted avg       0.81      0.79      0.80        624



# 🚀 How to Run

Clone the repository:

git clone https://github.com/Rajat083/Pneumonia-X-Ray-Detection.git
cd pneumonia-detection

Install dependencies:

pip install -r requirements.txt

Run the Jupyter notebook:

jupyter notebook pneumonia_detection.ipynb
# 📚 References

Kaggle: Chest X-Ray Images (Pneumonia)

Rajpurkar et al. (2017), CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.

Chollet, F. (2017). Deep Learning with Python.