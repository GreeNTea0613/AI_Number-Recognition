from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import base64
import io
import tensorflow as tf
import cv2

app = Flask(__name__)
CORS(app)
model = tf.keras.models.load_model("model_1000.keras")

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img.save("debug_1_raw.png")

    img = ImageOps.grayscale(img)
    img.save("debug_2_gray.png")

    img_np = np.array(img)

    _, img_np = cv2.threshold(img_np, 180, 255, cv2.THRESH_BINARY_INV)
    Image.fromarray(img_np).save("debug_3_binary.png")

    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        Image.fromarray(img_np).save("debug_4_no_contours.png")
        return None

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    digit = img_np[y:y + h, x:x + w]
    Image.fromarray(digit).save("debug_5_cropped.png")

    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    pad_x = (size - w) // 2
    pad_y = (size - h) // 2
    square[pad_y:pad_y + h, pad_x:pad_x + w] = digit
    Image.fromarray(square).save("debug_6_square.png")

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    Image.fromarray(resized).save("debug_7_resized.png")

    normalized = resized.astype("float32") / 255.0
    normalized = normalized.reshape(1, 28, 28, 1)

    return normalized


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img_data = data.get("image")

    if img_data is None:
        return jsonify({"error": "no image data"}), 400
    try:
        img_binary = base64.b64decode(img_data.split(",")[1])
    except Exception:
        return jsonify({"error": "invalid image format"}), 400

    img = Image.open(io.BytesIO(img_binary))

    processed = preprocess_image(img)
    if processed is None:
        return jsonify({"error": "empty image"})

    pred = model.predict(processed)
    pred = pred[0]

    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return jsonify({
        "digit": digit,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
