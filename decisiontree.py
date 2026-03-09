# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import ML model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# Example dataset
X = [[2,4],
     [3,4],
     [4,5],
     [6,7],
     [7,8],
     [8,10]]

y = [0,0,0,1,1,1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Create Decision Tree model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test,y_pred))

# Plot Decision Tree
plt.figure(figsize=(10,6))
plot_tree(model, filled=True)
plt.show()
