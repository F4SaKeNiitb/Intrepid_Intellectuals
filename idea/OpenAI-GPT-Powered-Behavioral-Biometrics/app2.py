import streamlit as st
import streamlit.components.v1 as components
import json
import time
import numpy as np
import pickle

# Load your trained model
try:
    with open('keystroke_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
except FileNotFoundError:
    st.error("The model file 'keystroke_model.pkl' was not found.")
    st.stop()

# Constants for keystroke features
KEY_HOLD = 'hold_time'
KEY_RELEASE = 'release_time'
TYPING_SPEED = 'typing_speed'

def extract_features(keystrokes):
    """
    Process the keystroke list to extract features.
    Assumes keystrokes is a list of dicts like:
      { "key": "a", "event": "down", "time": timestamp }
    """
    features = []
    key_down_times = {}  # Track when each key is pressed
    previous_key_up_time = None

    for entry in keystrokes:
        key = entry.get("key")
        event_type = entry.get("event")
        timestamp = entry.get("time") / 1000  # converting milliseconds to seconds
        if event_type == "down":
            key_down_times[key] = timestamp
        elif event_type == "up":
            if key in key_down_times:
                hold_time = timestamp - key_down_times[key]
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
                del key_down_times[key]
    return features

def authenticate_user(model, keystroke_features):
    # Convert features to the appropriate format for prediction
    X = np.array([[kf[KEY_HOLD], kf[KEY_RELEASE], kf[TYPING_SPEED]] for kf in keystroke_features])
    prediction = model.predict(X)
    return prediction[0]

# Title and instructions
st.title("Keystroke Authentication App")
st.write("Type in the box below. Your keystrokes will be recorded for 10 seconds.")

# HTML/JS component that captures keystrokes.
# This code sets up a simple textarea and listens for key events.
html_code = """
<html>
  <head>
    <script>
      let keystrokes = [];
      // Capture keydown events
      document.addEventListener('keydown', function(e) {
        keystrokes.push({key: e.key, event: 'down', time: performance.now()});
      });
      // Capture keyup events
      document.addEventListener('keyup', function(e) {
        keystrokes.push({key: e.key, event: 'up', time: performance.now()});
      });
      // After 10 seconds, send the keystroke data to Streamlit
      setTimeout(function(){
        const keystrokeData = JSON.stringify(keystrokes);
        // The following sends data back to Streamlit.
        // It sets a hidden <textarea> value so that Streamlit can read it.
        document.getElementById("keystroke-data").value = keystrokeData;
      }, 10000);
    </script>
  </head>
  <body>
    <textarea id="inputBox" style="width:100%; height:200px;" placeholder="Start typing here..."></textarea>
    <!-- Hidden field to store the keystroke data -->
    <textarea id="keystroke-data" style="display:none;"></textarea>
  </body>
</html>
"""

# Render the component
result = components.html(html_code, height=300)

st.write("Wait for 10 seconds for the recording to finish, then click the button below.")

# A button to retrieve and process the keystroke data from the hidden textarea
if st.button("Process Keystroke Data"):
    # Use streamlit's JavaScript evaluation capability to read the content of the hidden textarea.
    # Note: This is a workaround. In a full custom component, you'd use Streamlit's proper component API.
    keystroke_data = st.experimental_get_query_params().get("keystroke-data", None)
    
    # Alternatively, if you can use a text input to paste the data (for testing), you might do:
    keystroke_data_input = st.text_area("Paste keystroke JSON data here (if not auto-filled):")
    if not keystroke_data and keystroke_data_input:
        keystroke_data = keystroke_data_input
    
    if keystroke_data:
        try:
            keystrokes = json.loads(keystroke_data)
            st.write("Keystroke Data Captured:", keystrokes[:5], "...")
            test_features = extract_features(keystrokes)
            if len(test_features) > 5:
                predicted_numeric = authenticate_user(loaded_model, [test_features[5]])
                label_to_numeric = {'user1': 0, 'user2': 1}
                numeric_to_label = {v: k for k, v in label_to_numeric.items()}
                st.success("Authenticated User: " + numeric_to_label[predicted_numeric])
            else:
                st.warning("Not enough keystroke data was captured. Please try again.")
        except Exception as e:
            st.error(f"Error processing keystroke data: {e}")
    else:
        st.info("Keystroke data not found. Please ensure you typed in the text box and wait for 10 seconds.")
