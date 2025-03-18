import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keyboard
import pickle


# Constants for keystroke features
KEY_HOLD = 'hold_time'
KEY_RELEASE = 'release_time'
TYPING_SPEED = 'typing_speed'

def collect_keystroke_data(duration=10):
    print("Collecting keystroke data for User...")
    keystrokes = []
    start_time = time.time()
    while time.time() - start_time < duration:
        event = keyboard.read_event(suppress=True)
        timestamp = time.time()
        keystrokes.append((event, timestamp))
    return keystrokes

def extract_features(keystrokes):
    """
    Extract features by pairing key down and key up events:
      - Hold time: time between key down and key up.
      - Release time: time interval between consecutive key up events.
      - Typing speed: inverse of the release time.
      
    Note: This simplified method assumes that keys are pressed and released sequentially.
    """
    features = []
    key_down_times = {}  # Track when each key is pressed
    previous_key_up_time = None

    for event, timestamp in keystrokes:
        if event.event_type == keyboard.KEY_DOWN:
            # Record the time of the key down event
            key_down_times[event.name] = timestamp
        elif event.event_type == keyboard.KEY_UP:
            # Calculate hold time if we have a matching key down event
            if event.name in key_down_times:
                hold_time = timestamp - key_down_times[event.name]
                # Calculate release time (interval between consecutive key ups)
                if previous_key_up_time is not None:
                    release_time = timestamp - previous_key_up_time
                    typing_speed = 1 / release_time if release_time > 0 else 0
                else:
                    release_time = 0
                    typing_speed = 0

                features.append({
                    KEY_HOLD: hold_time,
                    KEY_RELEASE: release_time,
                    TYPING_SPEED: typing_speed
                })
                previous_key_up_time = timestamp
                del key_down_times[event.name]
    return features

def train_model(features, labels):
    # Prepare feature matrix
    X = np.array([[f[KEY_HOLD], f[KEY_RELEASE], f[TYPING_SPEED]] for f in features])
    
    # Use explicit mapping for labels
    label_to_numeric = {'user1': 0, 'user2': 1}
    y = np.array([label_to_numeric[label] for label in labels])
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)
    
    return model, label_to_numeric

def authenticate_user(model, keystroke_features):
    # Convert features to the appropriate format for prediction
    X = np.array([[kf[KEY_HOLD], kf[KEY_RELEASE], kf[TYPING_SPEED]] for kf in keystroke_features])
    prediction = model.predict(X)
    return prediction[0]

def main():
    # Data collection for two users
    print("User1: Please type for 10 seconds...")
    keystrokes1 = collect_keystroke_data(duration=30)
  
    
    features_user1 = extract_features(keystrokes1)

    
    # Create labels for each feature sample
    user1_labels = ['user1'] * len(features_user1)

    
    features = features_user1 
    user_labels = user1_labels 

    model, label_to_numeric = train_model(features, user_labels)
    
    
    print(f"label to numeric {label_to_numeric}")
    quit()
    
    with open('keystroke_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as keystroke_model.pkl")
    
    # Authentication phase: capture new keystroke data
    print("Authentication: Please type for 10 seconds...")
    keystrokes_test = collect_keystroke_data(duration=10)
    test_features = extract_features(keystrokes_test)
    
    if not test_features:
        print("No valid keystroke features extracted from test data.")
        return
    
    # For demonstration, use the first feature sample from the test data
    predicted_numeric = authenticate_user(model, [test_features[0]])
    # Invert the mapping to get the label name
    numeric_to_label = {v: k for k, v in label_to_numeric.items()}
    print("Authenticated User:", numeric_to_label[predicted_numeric])

if __name__ == "__main__":
    main()
