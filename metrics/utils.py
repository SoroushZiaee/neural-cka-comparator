import numpy as np

def normalize(X):
    return (X - X.mean()) / X.std()

# Add any other utility functions related to metrics calculations here