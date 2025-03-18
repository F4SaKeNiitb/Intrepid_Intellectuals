import app as st
import time
import numpy as np
import keyboard
import pickle

# Constants for keystroke features
KEY_HOLD = 'hold_time'
KEY_RELEASE = 'release_time'
TYPING_SPEED = 'typing_speed'

def collect_keystroke_data(duration=10):
    st.write("Collecting keystroke data for the user...")
    keystrokes = []
    start_time = time.time()
    while time.time() - start_time < duration:
        # This blocks until an event is detected.
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
      
    This simplified method assumes that keys are pressed and released sequentially.
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

def authenticate_user(model, keystroke_features):
    # Convert features to the appropriate format for prediction
    X = np.array([[kf[KEY_HOLD], kf[KEY_RELEASE], kf[TYPING_SPEED]] for kf in keystroke_features])
    prediction = model.predict(X)
    return prediction[0]

# Mapping for labels
label_to_numeric = {'user1': 0, 'user2': 1}
numeric_to_label = {v: k for k, v in label_to_numeric.items()}

# Load the trained model from file
try:
    with open('keystroke_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    st.error("The model file 'keystroke_model.pkl' was not found.")
    st.stop()

st.title("Keystroke Authentication App")
st.write("Click the button below to start authentication. Please type for 10 seconds when prompted.")

if st.button("Start Authentication"):
    st.info("Please start typing for 10 seconds. Your keystrokes are being recorded...")
    keystrokes_test = collect_keystroke_data(duration=10)
    test_features = extract_features(keystrokes_test)
    
    # Check if we have collected enough features; using the 6th feature as in your sample code.
    if len(test_features) > 5:
        predicted_numeric = authenticate_user(loaded_model, [test_features[5]])
        st.success("Authenticated User: " + numeric_to_label[predicted_numeric])
    else:
        st.warning("Not enough keystroke data was captured. Please try again.")
