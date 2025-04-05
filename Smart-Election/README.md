# Smart Election System

A facial recognition-based voting system that prevents duplicate voting.

## System Requirements
- Python 3.6+
- Webcam
- Windows OS (for pywin32 text-to-speech)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup Instructions

### 1. Register Voters
Run the face registration system:
```bash
python add_faces.py
```
- Enter Aadhar number when prompted
- Face the camera until 51 frames are captured (about 10-15 seconds)
- Press 'q' to exit early if needed

### 2. Start Voting System
Run the voting application:
```bash
python give_vote.py
```

## Voting Process

1. Face the camera for recognition
2. When your Aadhar number appears:
   - Press '1' to vote for BJP
   - Press '2' to vote for Congress  
   - Press '3' to vote for AAP
   - Press '4' to vote for NOTA
3. The system will:
   - Announce your vote is recorded
   - Save to Votes.csv
   - Prevent duplicate voting

## Output Files

- `data/faces_data.pkl` - Registered face embeddings
- `data/names.pkl` - Registered Aadhar numbers
- `data/voted_faces.pkl` - Faces that have voted
- `Votes.csv` - Voting records with timestamps

## Security Features

- Prevents voting with same Aadhar number
- Prevents voting with same face (even with different Aadhar)
- Maintains complete voting audit trail

## Troubleshooting

**Webcam not working:**
- Check if camera is connected
- Ensure no other apps are using camera
- Try different USB port if available

**Face not recognized:**
- Ensure proper lighting
- Face camera directly
- Remove glasses/hat if worn during registration
