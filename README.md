# Handwritten Digit Recognition using CNN

An end-to-end handwritten digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project includes a robust image preprocessing pipeline and a Tkinter-based GUI for real-time digit prediction with confidence scores.

---

## 📌 Project Overview
This project demonstrates how deep learning can be applied to recognize handwritten digits (0–9). A CNN model is trained on the MNIST dataset and deployed in a graphical interface where users can draw digits and instantly receive predictions.

---

## 🚀 Features
- CNN-based digit classification (0–9)
- ~99% test accuracy on MNIST dataset
- Real-time handwritten digit recognition
- Interactive Tkinter GUI
- Complete preprocessing pipeline for real-world inputs
- End-to-end workflow: training → evaluation → deployment

---

## 🧠 Model Architecture
- Conv2D + ReLU
- MaxPooling
- Dropout (regularization)
- Fully Connected Dense layers
- Softmax output layer

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Epochs:** 10  
**Test Accuracy:** ~99.18%

---

## 📊 Training Results

The model shows steady improvement in accuracy and reduction in loss across epochs.

<p align="center">
  <img src="screenshots/training_logs.png" width="700"/>
</p>

---

## 🖥️ GUI Output Samples

The GUI allows users to draw digits and displays the predicted digit along with confidence scores.

<p align="center">
  <img src="outputs/prediction_0.png" width="220"/>
  <img src="outputs/prediction_5.png" width="220"/>
</p>

<p align="center">
  <img src="outputs/prediction_3.png" width="220"/>
  <img src="outputs/prediction_7.png" width="220"/>
</p>

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Tkinter
- Pillow (PIL)

---

## 📂 Project Structure
