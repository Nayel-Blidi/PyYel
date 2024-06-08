
import cv2
import torch

from pyl.models.vision.detectionssd import DetectionSSD
from pyl.models.vision.detectionfasterrcnn import DetectionFasterRCNN

coco_labels = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush"
}


def pyl_api_demo():
    """
    PyYel API demosntration. Loads a pretrained FasterRCNNMobileNet320 model and returns
    it in inference mode.
    """

    # PyYel API init
    api = DetectionFasterRCNN(None, version="MobileNet320")

    # Model (custom/pretrained) loading
    api.load_model()

    # Inference mode
    model = api.model
    model.eval()

    return model

def main(frame_ratio:int=1, escape_key='q'):
    """
    Main loop. Captures PC camera flux and plots the video with inferred boxes & labels.
    """

    # This is PyYel
    model = pyl_api_demo()

    # This is no longer PyYel
    cap = cv2.VideoCapture(0)  # Use the default camera (index 0)
    frame_count = 0
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_ratio == 0:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)

            with torch.no_grad():
                predictions = model(input_tensor.unsqueeze(0))

            for pred in predictions:
                for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):

                    if score >= 0.5:
                        box = [int(coord) for coord in box]
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

                        label_name = coco_labels.get(label.item(), "Unknown")
                        label_text = f"{label_name}: {score:.2f}"
                        cv2.putText(frame, label_text, (box[0], box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord(escape_key):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(frame_ratio=2)
