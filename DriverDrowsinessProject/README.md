# Driver Drowsiness Detection 🚗💤

## 📌 Description
This project detects driver drowsiness in real-time using computer vision. It monitors eye movements and alerts the driver using sound and SMS.

## 🚀 Features
- Real-time eye tracking
- Drowsiness detection
- Alarm alert system
- Emergency SMS using Twilio

## 🛠️ Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- Twilio API

## ▶️ How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the project:
python drowsiness_proto.py

## 🔐 Environment Variables
Create a `.env` file and add:

TWILIO_ACCOUNT_SID=your_sid  
TWILIO_AUTH_TOKEN=your_token

## 📂 Project Files
- drowsiness_proto.py → Main code  
- alert.wav → Alarm sound  
- CSV files → Logs  