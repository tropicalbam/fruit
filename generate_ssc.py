import os
import numpy as np
import librosa
import joblib
from sklearn.neural_network import MLPRegressor
import traceback

# Path to the trained model
model_path = r'D:\DDRAI\trained_model.joblib'

# Directory to save the generated SSC files
output_directory = r'D:\[step 2] validation_data\madebyai'

# Directory containing the audio files
audio_files_directory = r'D:\[step 2] validation_data\audio_files2'

# Load the model
print("Loading model...")
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully: {type(model)}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

def extract_features(audio_file):
    try:
        print(f"Extracting features from {audio_file}...")
        y, sr = librosa.load(audio_file, sr=None)
        print(f"Audio loaded. Length: {len(y)}, Sample rate: {sr}")
        
        if len(y) == 0:
            print(f"Error: Audio file {audio_file} is empty.")
            return None, None, None
        
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Duration: {duration} seconds")
        
        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Create a dictionary of features
        features = {
            'tempo': tempo,
            'mfccs_mean': np.mean(mfccs, axis=1),
            'mfccs_var': np.var(mfccs, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_var': np.var(chroma, axis=1)
        }
        
        print(f"Extracted features: {list(features.keys())}")
        return features, tempo, duration
    except Exception as e:
        print(f"Failed to extract features from {audio_file}: {e}")
        traceback.print_exc()
        return None, None, None

def predict_notes(model, features, duration, tempo):
    try:
        # Flatten and combine all features
        feature_vector = np.concatenate([
            [features['tempo']],
            features['mfccs_mean'],
            features['mfccs_var'],
            features['chroma_mean'],
            features['chroma_var']
        ])
        
        # Reshape features to match the model's expected input
        features_reshaped = feature_vector.reshape(1, -1)
        print(f"Reshaped features shape: {features_reshaped.shape}")
        print(f"Model's expected input shape: {model.n_features_in_}")
        
        if features_reshaped.shape[1] != model.n_features_in_:
            print(f"Warning: Number of features ({features_reshaped.shape[1]}) doesn't match model's expected input ({model.n_features_in_})")
            # Pad or truncate features to match model's expected input
            if features_reshaped.shape[1] < model.n_features_in_:
                features_reshaped = np.pad(features_reshaped, ((0, 0), (0, model.n_features_in_ - features_reshaped.shape[1])))
            else:
                features_reshaped = features_reshaped[:, :model.n_features_in_]
        
        # Predict notes
        raw_predictions = model.predict(features_reshaped)
        
        # Calculate number of steps based on duration and tempo
        steps = int(duration * tempo / 60)
        
        # Ensure we have the correct number of predictions
        if len(raw_predictions) < steps:
            raw_predictions = np.pad(raw_predictions, (0, steps - len(raw_predictions)))
        elif len(raw_predictions) > steps:
            raw_predictions = raw_predictions[:steps]
        
        return raw_predictions
    except Exception as e:
        print(f"Error in predicting notes: {e}")
        traceback.print_exc()
        return None

def map_to_ddr_note(step_values):
    # Ensure we only have one or two arrows pressed at a time
    active_arrows = np.argsort(step_values)[-2:]  # Get indices of the two highest values
    mapped_step = ['0'] * 4
    for i in active_arrows:
        if step_values[i] > 0.5:  # Only activate if the value is high enough
            mapped_step[i] = '1'
    return ''.join(mapped_step)

def format_notes(predicted_notes, difficulty, beats_per_measure=16):
    formatted_notes = ""
    for i in range(0, len(predicted_notes), beats_per_measure):
        measure = predicted_notes[i:i+beats_per_measure]
        mapped_measure = [map_to_ddr_note(step) for step in measure]
        
        # Adjust difficulty
        if difficulty == 'Beginner':
            mapped_measure = ['0000' if j % 4 != 0 else step for j, step in enumerate(mapped_measure)]
        elif difficulty == 'Easy':
            mapped_measure = ['0000' if j % 2 != 0 else step for j, step in enumerate(mapped_measure)]
        
        formatted_notes += "\n".join(mapped_measure) + "\n,  \n"
    return formatted_notes.strip()

def generate_ssc_from_chart(audio_file, predicted_notes, tempo, duration):
    difficulties = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
    ssc_template = f"""#VERSION:0.83;
#TITLE:Generated by StepGPT;
#ARTIST:StepGPT;
#MUSIC:{os.path.basename(audio_file)};
#OFFSET:0.009;
#SAMPLESTART:54.548077;
#SAMPLELENGTH:11.550000;
#SELECTABLE:YES;
#BPMS:0.000={tempo};
#STOPS:;
#DELAYS:;
#WARPS:;
#TIMESIGNATURES:0.000=4=4;
#TICKCOUNTS:0.000=4;
#COMBOS:0.000=1;
#SPEEDS:0.000=1.000=0.000=0;
#SCROLLS:0.000=1.000;
#FAKES:;
#LABELS:0.000=Song Start;
#BGCHANGES:;
#KEYSOUNDS:;
#ATTACKS:
;
"""

    for difficulty in difficulties:
        meter = {'Beginner': 1, 'Easy': 4, 'Medium': 7, 'Hard': 9, 'Challenge': 12}[difficulty]
        formatted_notes = format_notes(predicted_notes, difficulty)
        ssc_template += f"""#NOTEDATA:;
#CHARTNAME:StepGPT Chart;
#STEPSTYPE:dance-single;
#DESCRIPTION:;
#CHARTSTYLE:;
#DIFFICULTY:{difficulty};
#METER:{meter};
#RADARVALUES:0.491635,0.531859,0.000000,0.231747,0.034762,297.000000,297.000000,0.000000,20.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.491635,0.531859,0.000000,0.231747,0.034762,297.000000,297.000000,0.000000,20.000000,0.000000,0.000000,0.000000,0.000000,0.000000;
#CREDIT:StepGPT;
#NOTES:
{formatted_notes}
;
"""

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Create the path for the generated SSC file
    generated_ssc_file = os.path.join(output_directory, os.path.basename(audio_file).replace('.ogg', '_generated.ssc'))
    
    try:
        with open(generated_ssc_file, 'w') as file:
            file.write(ssc_template)
        print(f'SSC file generated for {audio_file}: {generated_ssc_file}')
    except Exception as e:
        print(f"Failed to write SSC file for {audio_file}: {e}")

# Main execution
# Get list of all OGG files in the audio_files_directory
try:
    print(f"Listing files in directory: {audio_files_directory}")
    all_files = os.listdir(audio_files_directory)
    print(f"All files: {all_files}")
    test_audio_files = [os.path.join(audio_files_directory, f) for f in all_files if f.lower().endswith('.ogg')]
    print(f"Found {len(test_audio_files)} audio files.")
except Exception as e:
    print(f"Failed to list audio files: {e}")
    exit(1)

# Loop through the test files and generate SSC files
for audio_file in test_audio_files:
    try:
        if not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} does not exist.")
            continue
        
        print(f"Processing {audio_file}...")
        features, tempo, duration = extract_features(audio_file)
        if features is not None:
            predicted_notes = predict_notes(model, features, duration, tempo)
            if predicted_notes is not None:
                generate_ssc_from_chart(audio_file, predicted_notes, tempo, duration)
            else:
                print(f"Skipping {audio_file} due to failed note prediction.")
        else:
            print(f"Skipping {audio_file} due to failed feature extraction.")
    except Exception as e:
        print(f"Failed to process {audio_file}: {e}")
        traceback.print_exc()

print("Processing complete.")