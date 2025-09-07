"""
Test how a classifier model works on a dataset
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    # Load model
    model = load_model("models/eye_color_v0.h5")

    # Your class mapping (make sure this matches train_gen.class_indices)
    idx_to_class = {0:"amber", 1:"blue", 2:"brown", 3:"gray", 4:"green", 5:"hazel"}

    IMG_SIZE = (128, 128)

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame (resize + normalize)
        resized = cv2.resize(frame, IMG_SIZE)
        norm = resized / 255.0
        inp = np.expand_dims(norm, axis=0)

        # Predict
        pred = model.predict(inp, verbose=0)
        pred_class = idx_to_class[np.argmax(pred)]
        confidence = np.max(pred)

        # Display prediction
        cv2.putText(frame, f"{pred_class} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Eye Color Detection", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
