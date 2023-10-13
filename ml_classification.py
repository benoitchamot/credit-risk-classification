# This module contains functions that are commonly used in classification projects (supervised learning)
# ---
# Author: Benoit Chamot
# Date: 13/10/2023
# ---

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def model_performance(y_test, predictions):
# Print the accuracy, confusion matrix and classification report
    # Calculating the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(
        cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
    )

    # Calculating the accuracy score
    acc_score = accuracy_score(y_test, predictions)

    # Displaying results
    print("Confusion Matrix:")
    print(cm_df)
    print('---')
    print(f"Accuracy Score : {acc_score}")
    print('---')
    print("Classification Report:")
    print(classification_report(y_test, predictions))