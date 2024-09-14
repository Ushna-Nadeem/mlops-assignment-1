import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# A small fun dataset with exactly 50 samples
def generate_dataset():
    # Ingredients: flour (g), sugar (g), salt (g), butter (g)
    data = [
        [200, 100, 1, 50],  # Sweet
        [300, 0, 5, 60],    # Savory
        [250, 150, 2, 55],  # Sweet
        [150, 50, 3, 70],   # Savory
        [180, 200, 0.5, 80],# Sweet
        [220, 0, 6, 45],    # Savory
        [175, 130, 1, 65],  # Sweet
        [200, 10, 5, 60],   # Savory
        [160, 180, 0.2, 40],# Sweet
        [300, 0, 8, 70],    # Savory
        [190, 120, 0.5, 55],# Sweet
        [220, 0, 7, 60],    # Savory
        [180, 140, 1, 45],  # Sweet
        [250, 0, 4, 80],    # Savory
        [230, 110, 0.3, 50],# Sweet
        [280, 0, 5, 65],    # Savory
        [240, 90, 0.5, 55], # Sweet
        [260, 0, 7, 75],    # Savory
        [200, 130, 0.7, 40],# Sweet
        [210, 0, 6, 70],    # Savory
        [150, 160, 0.4, 50],# Sweet
        [270, 0, 8, 80],    # Savory
        [190, 140, 0.6, 45],# Sweet
        [300, 0, 7, 60],    # Savory
        [230, 120, 0.5, 55],# Sweet
        [210, 0, 6, 65],    # Savory
        [240, 100, 1, 50],  # Sweet
        [280, 0, 5, 75],    # Savory
        [200, 150, 0.8, 40],# Sweet
        [220, 0, 7, 70],    # Savory
        [170, 130, 1, 60],  # Sweet
        [250, 0, 5, 80],    # Savory
        [160, 110, 0.3, 55],# Sweet
        [270, 0, 6, 45],    # Savory
        [180, 140, 0.4, 50],# Sweet
        [230, 0, 8, 70],    # Savory
        [190, 120, 0.7, 40],# Sweet
        [220, 0, 6, 75],    # Savory
        [210, 100, 0.5, 60],# Sweet
        [260, 0, 5, 65],    # Savory
        [200, 140, 1, 55],  # Sweet
        [240, 0, 7, 50],    # Savory
        [230, 130, 0.2, 45],# Sweet
        [300, 0, 6, 80],    # Savory
        [170, 110, 0.4, 55],# Sweet
        [210, 0, 8, 60],    # Savory
        [250, 90, 0.6, 40], # Sweet
        [222, 0, 9, 61],    # Savory
        [250, 70, 0.6, 30], # Sweet
        [300, 0, 6, 50],    # Savory
    ]

    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    return np.array(data), np.array(labels)

def train_and_save_model():
    # Generate dataset
    X, y = generate_dataset()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Save the model to a file
    joblib.dump(model, 'model.pkl')
    print("Model saved as 'model.pkl'.")

if __name__ == "__main__":
    train_and_save_model()
