import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

# Training data
X = [[30], [40], [50], [60], [20], [10], [70]]
y = [0, 1, 1, 1, 0, 0, 1]

# Create and train the SVM classifier
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X, y)

# Test data
X_marks = [[55]]

# Prediction
y_pred = classifier.predict(X_marks)

# Output
print("Predicted class:", y_pred)