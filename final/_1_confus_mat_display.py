# Importing required libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Example data (replace with actual data)
X = np.random.rand(100, 10)  # Replace with actual features
y = np.random.randint(0, 5, 100)  # Replace with actual target labels (5 classes)

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (e.g., Naive Bayes)
model = GaussianNB()
model.fit(x_train, y_train)

# Make predictions
y_pred_nb = model.predict(x_test)

# Matrix display function
def display_confus_matrix(actual, pred):
    confus_matrix = confusion_matrix(actual, pred)
    confus_matrix_dis = ConfusionMatrixDisplay(
        confusion_matrix=confus_matrix,
        display_labels=["Normal", "Blackhole", "TCP-SYN", "PortScan", "Diversion"],
    )
    confus_matrix_dis.plot()

# Main
if __name__ == "__main__":
    # Call the function with the actual and predicted values
    display_confus_matrix(y_test, y_pred_nb)
