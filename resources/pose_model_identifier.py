import pandas as pd

BODY_IDENTIFIERS = {
    "nose": 0,
    "neck": -1,
    "rightEye": 5,
    "leftEye": 2,
    "rightEar": 8,
    "leftEar": 7,
    "rightShoulder": 12,
    "leftShoulder": 11,
    "rightElbow": 14,
    "leftElbow": 13,
    "rightWrist": 16,
    "leftWrist": 15
}
HAND_IDENTIFIERS = {
    "wrist": 0,
    "indexTip": 8,
    "indexDIP": 7,
    "indexPIP": 6,
    "indexMCP": 5,
    "middleTip": 12,
    "middleDIP": 11,
    "middlePIP": 10,
    "middleMCP": 9,
    "ringTip": 16,
    "ringDIP": 15,
    "ringPIP": 14,
    "ringMCP": 13,
    "littleTip": 20,
    "littleDIP": 19,
    "littlePIP": 18,
    "littleMCP": 17,
    "thumbTip": 4,
    "thumbIP": 3,
    "thumbMP": 2,
    "thumbCMC": 1
}


class mp_holistic_data:
    def __init__(self, column_names):
        self.data_hub = {}
        for n in column_names[1:-1]:
            self.data_hub[n] = []

    def hand_append_zero(self, handedness):
        for k in self.data_hub.keys():
            if "_" + handedness + "_" in k:
                self.data_hub[k].append(0)

    def hand_append_value(self, handedness, hand_landmarks):
        for name, lm_idx in HAND_IDENTIFIERS.items():
            lm = hand_landmarks.landmark[lm_idx]
            for xy, xy_value in zip(['_X', '_Y'], [lm.x, lm.y]):
                k = name + '_' + handedness + xy
                self.data_hub[k].append(xy_value)

    def get_series(self):
        return pd.Series(self.data_hub)

    def extract_data(self, holistic_results):
        def neck(pose_results):
            ls = pose_results.pose_landmarks.landmark[11]
            rs = pose_results.pose_landmarks.landmark[12]
            no = pose_results.pose_landmarks.landmark[0]
            if (ls.visibility > 0.5) & (rs.visibility > 0.5) & (no.visibility > 0.5):
                # This indicates the neck better. But it does not affect the result.
                cx = (ls.x + rs.x) / 2
                cy = (ls.y + rs.y) / 2
                dx = no.x - cx
                dy = no.y - cy
                x = cx + 0.3 * dx
                y = cy + 0.3 * dy
                # x = (ls.x+rs.x)/2
                # y = (ls.y+rs.y)/2
            else:
                x = 0
                y = 0
            return [x, y]

        # for the frame that can not extract skeleton from
        if not holistic_results.pose_landmarks:
            return
        for name, lm_idx in BODY_IDENTIFIERS.items():
            if name == "neck":
                xy_value = neck(holistic_results)
            else:
                lm = holistic_results.pose_landmarks.landmark[lm_idx]
                visible = float(lm.visibility >= 0.5)
                xy_value = [lm.x * visible, lm.y * visible]
            for xy_id, xy in zip(['_X', '_Y'], xy_value):
                s_name = name + xy_id
                self.data_hub[s_name].append(xy)

        for handedness, lm in zip(['Right', 'Left'],
                                  [holistic_results.right_hand_landmarks, holistic_results.left_hand_landmarks]):
            if lm:
                self.hand_append_value(handedness, lm)
            else:
                self.hand_append_zero(handedness)
        return