import pickle
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Initialize empty voted_faces.pkl
with open('data/voted_faces.pkl', 'wb') as f:
    pickle.dump([], f)
    
print("Successfully initialized voted_faces.pkl")
