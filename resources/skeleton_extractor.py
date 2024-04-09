# import warnings
import pandas as pd
from os import path
import cv2
import mediapipe as mp
import json
from resources.pose_model_identifier import BODY_IDENTIFIERS, HAND_IDENTIFIERS, mp_holistic_data

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic()

column_names = []
column_names.append('video_id')
for id_name in BODY_IDENTIFIERS.keys():
    for xy in ["_X", "_Y"]:
        column_names.append(id_name + xy)

for lr in ["_Right", "_Left"]:
    for id_name in HAND_IDENTIFIERS.keys():
        for xy in ["_X", "_Y"]:
            column_names.append(id_name + lr + xy)

column_names.append('labels')


def create_df(flnm, column_names):
    df = pd.DataFrame(columns=column_names)
    return df


def save_data(df, data, flnm):
    df = df.append(data.get_series(), ignore_index=True)
    df.to_pickle(flnm)


def obtain_pose_data(path):
    cap = cv2.VideoCapture(path)
    data = mp_holistic_data(column_names)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detection
        holistic_results = holistic.process(image)
        # Extract feature and save to mp_pose_data class
        data.extract_data(holistic_results)
    cap.release()

    return data


if __name__ == '__main__':
    pass

