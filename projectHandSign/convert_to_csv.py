import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hand module.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5)

# Define a function to extract hand keypoints from an image.
def extract_hand_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Extract keypoints for the first detected hand (assuming only one hand in each image).
        landmarks = results.multi_hand_landmarks[0].landmark
        keypoints = [landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks]
        return keypoints
    else:
        return None

# Define the path to the parent folder containing subfolders with hand photos.
parent_folder = 'data'

# Initialize empty lists to store data.
image_paths = []
keypoints_list = []

# Initialize a list to store DataFrames for each subfolder.
dataframes = []

# Iterate through subfolders in the parent folder.
for subfolder_name in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder_name)
    if os.path.isdir(subfolder_path):
        # Initialize lists for each subfolder.
        subfolder_image_paths = []
        subfolder_keypoints_list = []

        # Iterate through the image files in the subfolder.
        for image_filename in os.listdir(subfolder_path):
            if image_filename.endswith('.png'):
                image_path = os.path.join(subfolder_path, image_filename)
                keypoints = extract_hand_keypoints(image_path)
                if keypoints is not None:
                    subfolder_image_paths.append(image_path)
                    subfolder_keypoints_list.append(keypoints)

        # Create a Pandas DataFrame for the subfolder.
        df = pd.DataFrame(subfolder_keypoints_list, columns=[f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)])
        # Add a 'class' column with the subfolder name.
        df['class'] = subfolder_name
        dataframes.append(df)

# Concatenate DataFrames for all subfolders into a single DataFrame.
df_all = pd.concat(dataframes, ignore_index=True)

# Reorder columns to place the "class" column as the first column.
df_all = df_all[['class'] + [col for col in df_all.columns if col != 'class']]

# Save the DataFrame with data from all subfolders to a combined CSV file.
df_all.to_csv('all_hand_keypoints.csv', index=False)
