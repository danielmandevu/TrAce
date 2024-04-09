#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import gradio as gr
import mediapipe as mp
import os, whisper, csv, itertools
from keras.models import load_model
from PIL import Image


# In[2]:


import copy
import torch
from resources.skeleton_extractor import obtain_pose_data
from resources.normalization.body_normalization import normalize_single_dict as normalize_single_body_dict, BODY_IDENTIFIERS
from resources.normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict, HAND_IDENTIFIERS


# In[3]:


mp_hands = mp.solutions.hands
mp_hands_connections = mp.solutions.hands_connections
mp_draw = mp.solutions.drawing_utils 


# In[4]:


model = torch.load("trace.pth", map_location=torch.device('cpu')).train(False)
HAND_IDENTIFIERS = [id + "_Left" for id in HAND_IDENTIFIERS] + [id + "_Right" for id in HAND_IDENTIFIERS]
GLOSS = ['book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy', 'cousin', 'deaf', 'fine',
         'help', 'no', 'thin', 'walk', 'year', 'yes', 'all', 'black', 'cool', 'finish', 'hot', 'like', 'many', 'mother',
         'now', 'orange', 'table', 'thanksgiving', 'what', 'woman', 'bed', 'blue', 'bowling', 'can', 'dog', 'family',
         'fish', 'graduate', 'hat', 'hearing', 'kiss', 'language', 'later', 'man', 'shirt', 'study', 'tall', 'white',
         'wrong', 'accident', 'apple', 'bird', 'change', 'color', 'corn', 'cow', 'dance', 'dark', 'doctor', 'eat',
         'enjoy', 'forget', 'give', 'last', 'meet', 'pink', 'pizza', 'play', 'school', 'secretary', 'short', 'time',
         'want', 'work', 'africa', 'basketball', 'birthday', 'brown', 'but', 'cheat', 'city', 'cook', 'decide', 'full',
         'how', 'jacket', 'letter', 'medicine', 'need', 'paint', 'paper', 'pull', 'purple', 'right', 'same', 'son',
         'tell', 'thursday']


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:

    data_array = landmarks_tensor.numpy()
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index]

    return output


def dictionary_to_tensor(landmarks_dict: dict) -> torch.Tensor:

    output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)


# In[5]:


def gesture_translation(cam_feed):
    
    data = obtain_pose_data(cam_feed)
    
    depth_map = np.empty(shape=(len(data.data_hub["nose_X"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        depth_map[:, index, 0] = data.data_hub[identifier + "_X"]
        depth_map[:, index, 1] = data.data_hub[identifier + "_Y"]

    depth_map = torch.from_numpy(np.copy(depth_map))

    depth_map = tensor_to_dictionary(depth_map)

    keys = copy.copy(list(depth_map.keys()))
    for key in keys:
        data = depth_map[key]
        del depth_map[key]
        depth_map[key.replace("_Left", "_0").replace("_Right", "_1")] = data

    depth_map = normalize_single_body_dict(depth_map)
    depth_map = normalize_single_hand_dict(depth_map)

    keys = copy.copy(list(depth_map.keys()))
    for key in keys:
        data = depth_map[key]
        del depth_map[key]
        depth_map[key.replace("_0", "_Left").replace("_1", "_Right")] = data

    depth_map = dictionary_to_tensor(depth_map)

    depth_map = depth_map - 0.5

    inputs = depth_map.squeeze(0).to(device)
    outputs = model(inputs).expand(1, -1, -1)
    results = torch.nn.functional.softmax(outputs, dim=2).detach().numpy()[0,0]

    results = {GLOSS[i]: float(results[i]) for i in range(100)}

    return results


# In[6]:


model_w = whisper.load_model('base')


# In[7]:


def speech_text(aud):
    audio = whisper.load_audio(aud)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model_w.device)


    _, probs = model_w.detect_language(mel)
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model_w, mel, options)
    text = result.text.lower()
    return text[:-1]


# In[8]:


# #Static signs model loading ndi function to classfy them....
# alphabeta letazi
model_static = load_model("static_signs/keypoint_classifier.hdf5", compile=False)
with open('static_signs/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        class_name = [
            row[0] for row in keypoint_classifier_labels
        ]


# In[9]:


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


# In[10]:


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


# In[11]:


hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)


def final_static_predic(img):
    img = np.array(img)
    image = cv2.flip(img, 1)
    debug_image = copy.deepcopy(image)
    results = hands.process(image)
    if results.multi_hand_landmarks is not None:

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            print(results.multi_hand_landmarks)
        my_img = np.array(pre_processed_landmark_list)
        img_final = my_img.reshape(1,42)
        res = model_static.predict(img_final)
        index = np.argmax(np.squeeze(res))
        return class_name[index]
    else:
        return "Please try again."


# In[12]:


# final_static_predic(Image.open('imgs/a.jpg'))


# In[13]:


def gestures(word):
    gestures = {
        'eat':'resources/gestures/eat.mov',
        'help':'resources/gestures/help.mov',
        'decide':'resources/gestures/decide.mov',
        'hat':'resources/gestures/hat.mov'
    }
    video_path = gestures.get(word)
    return video_path


# In[14]:


with gr.Blocks(css="footer {visibility: hidden}",title='TrAce') as TrAce:
    gr.Markdown("<h1><center>SIGN LANGUAGE DICTIONARY</h1></center>")            
    with gr.Tab("Gestures"):
        with gr.Row():
            with gr.Column():
                cam_feed = gr.Video(source="webcam",label="Cam_Feed")
#                 search_btn = gr.Button(value="Search")
            with gr.Column():
                translated_gestur = gr.Label(num_top_classes=1, label="Text Transalation")
#Static signs translation          
    with gr.Tab("Static Signs"):
        with gr.Row():
            with gr.Column():
                static_sign_in = gr.Image(source="webcam",label="Static signs input")
            with gr.Column():
                static_result = gr.Label(label="Text Transalation")
                
#  Speech translation               
    with gr.Tab("Speech"):
        with gr.Row():
            in_audio = gr.Audio(source="microphone",  type="filepath", label='Record Voice')
        with gr.Row():
            with gr.Column():
                out_translation_en = gr.Textbox(label= 'Audio Translation')
            with gr.Column():
                spee_gesture = gr.Video(label = 'Sign language gesture')

    with gr.Tab("Text"):
        with gr.Row():
            with gr.Column():
                word = gr.Textbox(label="Search gesture")     
            with gr.Column():
                gesture_output = gr.Video(label="Sign language gesture")      
    cam_feed.change(gesture_translation, inputs=cam_feed, outputs=translated_gestur)
    static_sign_in.change(final_static_predic,static_sign_in,static_result) 
    word.change(gestures,word,gesture_output)
    in_audio.change(speech_text, inputs=[in_audio], outputs=[out_translation_en])
    out_translation_en.change(gestures,inputs=out_translation_en, outputs = spee_gesture)


# In[15]:


TrAce.launch()


# In[ ]:




