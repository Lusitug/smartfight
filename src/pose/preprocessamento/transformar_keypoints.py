import numpy as np

class TransformarKeypoints:
    # arrays
    @staticmethod
    def converter_float32(video_keypoints):
        video_keypoints = np.array(video_keypoints, dtype=np.float32)
        return video_keypoints

    # remove camada de pessoas detectadas [1] para melhorar estrutura
    @staticmethod
    def espremer_estrutura_keypoint(video_keypoints):
        if video_keypoints.shape[1] == 1:
            video_keypoints = np.squeeze(video_keypoints, axis=1)
        return video_keypoints
    # de: video_keypoints[frame][person][keypoint][coord] # para: video_keypoints[frame][keypoint][coord]
