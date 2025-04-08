import cv2
from labels_keypoints import nomes_juntas

def pre_process(frame):
    frame = cv2.resize(frame, (300, 650))
    return frame

nomes_juntas = nomes_juntas()

marca_largura = 115  
marca_altura = 30

def detectar_keypoints(frame, model):
    height, width, _ = frame.shape
    predicts = model(frame)
    frame_keypoints = []

    for predict in predicts:
        if predict.keypoints is not None:
            keypoints = predict.keypoints.xyn.cpu().numpy()
            
            for point in keypoints:
                keypoints_list = []
                for i in range(17):
                    x_norm, y_norm = point[i][0], point[i][1]

                    x_coord = int(x_norm * width)
                    y_coord = int(y_norm * height)
                    
                    keypoints_list.append((x_norm, y_norm))
                    
                    cv2.circle(frame, (x_coord, y_coord), radius=3, color=(0, 0, 255), thickness=-1)
                    cv2.putText(frame, nomes_juntas[i], (x_coord + 5, y_coord - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                frame_keypoints.append(keypoints_list)

    cv2.rectangle(    frame,    (width - marca_largura, height - marca_altura),  (width, height), (0, 0, 0), -1) # w 100 / h 25
    
    return frame, frame_keypoints