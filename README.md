House Price Prediction using Neural Networks
Project Overview
This project aims to create a neural network model that can predict house prices based on various features. The model uses a multilayer perceptron (MLP) architecture and is implemented using the Keras library in Python.
Dataset
The dataset used for this project is the California Housing dataset, which contains information about houses in California and their prices.
The dataset is split into training and testing sets (80% for training and 20% for testing).
Model Architecture
The model consists of the following layers:
Dense (64 units, activation='relu')
Dense (32 units, activation='relu')
Dense (1 unit)
The model is compiled with the Adam optimizer and mean squared error loss function.
Usage
Clone the repository: git clone https://github.com/mackimart1/house-price-prediction.git
Install required libraries: pip install -r requirements.txt
Run the model: python model.py
Evaluate the model: python evaluate.py
Requirements
Python 3.8+
TensorFlow 2.8+
Keras 2.8+
NumPy 1.20+
Matplotlib 3.5+ (for visualization)
Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request.
License
This project is licensed under the MIT License. See LICENSE for details.
Acknowledgments
Special thanks to the Keras and TensorFlow teams for their amazing libraries!