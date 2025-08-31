# smart_tourist_safety.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ================================
# PART 1: Anomaly Detection (Movement)
# ================================
class TouristAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.2, random_state=42)

    def train(self, data):
        """
        data: pandas DataFrame with features like speed, stop_time, direction_change
        """
        self.model.fit(data)

    def detect(self, new_data):
        """
        new_data: pandas DataFrame with same features
        returns: anomaly predictions (-1 = anomaly, 1 = normal)
        """
        return self.model.predict(new_data)


# ================================
# PART 2: NLP SOS Detection
# ================================
class SOSMessageDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()

    def train(self, texts, labels):
        """
        texts: list of messages
        labels: list of 0 (normal) or 1 (SOS)
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def detect(self, msg):
        """
        msg: string (tourist message)
        returns: 1 if SOS detected, else 0
        """
        X = self.vectorizer.transform([msg])
        return self.model.predict(X)[0]


# ================================
# DEMO USAGE
# ================================
if __name__ == "__main__":
    # ----- Movement anomaly detection -----
    movement_data = pd.DataFrame({
        "speed": [1.2, 1.1, 1.3, 5.5, 1.0, 0.9, 12.0],   # sudden high speed = anomaly
        "stop_time": [2, 3, 1, 0, 2, 3, 0]               # long stop = anomaly
    })

    anomaly_detector = TouristAnomalyDetector()
    anomaly_detector.train(movement_data)
    preds = anomaly_detector.detect(movement_data)
    print("\n[Movement Anomaly Detection Results]")
    print(pd.DataFrame({"speed": movement_data["speed"], 
                        "stop_time": movement_data["stop_time"],
                        "prediction": preds}))

    # ----- SOS message detection -----
    texts = [
        "Help me please", "I am in danger", "Where is the hotel",
        "Call ambulance", "Book a ticket"
    ]
    labels = [1, 1, 0, 1, 0]  # 1 = SOS, 0 = Normal

    sos_detector = SOSMessageDetector()
    sos_detector.train(texts, labels)

    test_msgs = ["I am stuck in forest", "What time is dinner", "Need rescue immediately"]
    print("\n[SOS Message Detection Results]")
    for msg in test_msgs:
        result = sos_detector.detect(msg)
        print(f"Message: '{msg}' → {'SOS Detected 🚨' if result == 1 else 'Normal ✅'}")
