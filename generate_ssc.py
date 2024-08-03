import os
import numpy as np
import librosa
import joblib
from sklearn.neural_network import MLPRegressor
import traceback
import random
import math

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
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)  # Reduce to 7 MFCCs
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Create a dictionary of features
        features = np.concatenate([
            np.array([tempo]).reshape(1, -1),
            np.mean(mfccs, axis=1).reshape(1, -1),
            np.mean(chroma, axis=1).reshape(1, -1)
        ], axis=1)
        
        features = features.flatten()  # Flatten the array to 1D
        
        print(f"Extracted features shape: {features.shape}")
        return features, tempo, duration
    except Exception as e:
        print(f"Failed to extract features from {audio_file}: {e}")
        traceback.print_exc()
        return None, None, None

def predict_notes(model, features, duration, tempo):
    try:
        features_reshaped = features.reshape(1, -1)
        print(f"Reshaped features shape: {features_reshaped.shape}")
        print(f"Model's expected input shape: {model.n_features_in_}")
        
        if features_reshaped.shape[1] != model.n_features_in_:
            print(f"Warning: Number of features ({features_reshaped.shape[1]}) doesn't match model's expected input ({model.n_features_in_})")
            if features_reshaped.shape[1] < model.n_features_in_:
                features_reshaped = np.pad(features_reshaped, ((0, 0), (0, model.n_features_in_ - features_reshaped.shape[1])))
            else:
                features_reshaped = features_reshaped[:, :model.n_features_in_]
        
        # Predict notes
        raw_predictions = model.predict(features_reshaped)
        
        # Ensure raw_predictions is a 1D array
        raw_predictions = raw_predictions.flatten()
        
        # Calculate number of steps based on duration and tempo
        steps = int(duration * tempo / 60 * 4)  # Assuming 4 beats per measure
        
        # Interpolate predictions to match the number of steps
        predicted_notes = np.interp(np.linspace(0, len(raw_predictions) - 1, steps), 
                                    np.arange(len(raw_predictions)), 
                                    raw_predictions)
        
        return predicted_notes
    except Exception as e:
        print(f"Error in predicting notes: {e}")
        traceback.print_exc()
        return None

def format_notes(predicted_notes, difficulty, tempo, duration):
    note_densities = {'Beginner': 0.25, 'Easy': 0.5, 'Medium': 0.75}
    note_density = note_densities[difficulty]
    formatted_notes = ""
    
    beats_per_second = tempo / 60
    total_beats = int(duration * beats_per_second)
    total_measures = math.floor(total_beats / 4)  # Assuming 4/4 time signature
    
    intro_measures = 4
    outro_measures = 2
    main_measures = total_measures - intro_measures - outro_measures

    # Simple patterns for each difficulty
    patterns = {
        'Beginner': [
            ['1000', '0000', '0000', '0000'],
            ['0100', '0000', '0000', '0000'],
            ['0010', '0000', '0000', '0000'],
            ['0001', '0000', '0000', '0000']
        ],
        'Easy': [
            ['1000', '0000', '0100', '0000'],
            ['0010', '0000', '0001', '0000'],
            ['1000', '0000', '0001', '0000'],
            ['0100', '0000', '0010', '0000']
        ],
        'Medium': [
            ['1000', '0100', '0010', '0001'],
            ['1000', '0001', '0100', '0010'],
            ['0100', '0010', '1000', '0001'],
            ['0010', '0001', '1000', '0100']
        ]
    }

    # Add intro measures
    for _ in range(intro_measures):
        formatted_notes += "0000\n0000\n0000\n0000\n,\n"

    # Generate notes for the main part of the song
    note_index = 0
    for _ in range(main_measures):
        if random.random() < note_density:
            pattern = random.choice(patterns[difficulty])
            formatted_notes += "\n".join(pattern) + "\n,\n"
        else:
            formatted_notes += "0000\n0000\n0000\n0000\n,\n"
        note_index += 4

    # Add outro measures
    for _ in range(outro_measures):
        formatted_notes += "0000\n0000\n0000\n0000\n,\n"

    return formatted_notes.strip()

def generate_ssc_from_chart(audio_file, predicted_notes, tempo, duration):
    difficulties = [
        ('Beginner', 1),
        ('Easy', 2),
        ('Medium', 4)
    ]
    
    # Round the tempo to the nearest whole number and convert to int
    rounded_tempo = int(np.round(tempo))
    
    ssc_template = f"""#VERSION:0.83;
#TITLE:Generated by StepGPT;
#ARTIST:StepGPT;
#MUSIC:{os.path.basename(audio_file)};
#OFFSET:0.009;
#SELECTABLE:YES;
#BPMS:0.000={rounded_tempo}.000;
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

    for difficulty, meter in difficulties:
        formatted_notes = format_notes(predicted_notes, difficulty, rounded_tempo, duration)
        ssc_template += f"""#NOTEDATA:;
#CHARTNAME:StepGPT Chart;
#STEPSTYPE:dance-single;
#DESCRIPTION:;
#CHARTSTYLE:;
#DIFFICULTY:{difficulty};
#METER:{meter};
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
if __name__ == "__main__":
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
