import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
X = [[70],[80],[40],[60],[50],[39],[90]]
y = [0,1,1,1,0,0,1]
classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X,y)
X_marks=[[36]]
print(classifier.predict(X_marks))