import tkinter as tk
from PIL import Image, ImageDraw
import io
import base64
import requests

# Flask API
API_URL = "http://127.0.0.1:5000/predict"

root = tk.Tk()
root.title("手書き数字を判定しますわーーー")

CANVAS_SIZE = 300
BACKGROUND_COLOR = "white"
DRAW_COLOR = "black"
BRUSH_RADIUS = 4

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg=BACKGROUND_COLOR
)
canvas.pack(padx=10, pady=10)

draw_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), BACKGROUND_COLOR)
draw = ImageDraw.Draw(draw_image)

def draw_line(event):
    x, y = event.x, event.y
    r = BRUSH_RADIUS

    canvas.create_oval(x - r, y - r, x + r, y + r,
                       fill=DRAW_COLOR, outline=DRAW_COLOR)

    draw.ellipse([x - r, y - r, x + r, y + r],
                 fill=DRAW_COLOR, outline=DRAW_COLOR)

canvas.bind("<B1-Motion>", draw_line)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=BACKGROUND_COLOR)

clear_btn = tk.Button(root, text="クリア", command=clear_canvas)
clear_btn.pack(pady=5)

def predict_digit():
    buffer = io.BytesIO()
    draw_image.save(buffer, format="PNG")
    img_base64 = "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()

    try:
        response = requests.post(API_URL, json={"image": img_base64})
        response.raise_for_status()
        result = response.json()

        if "digit" in result:
            digit = result["digit"]
            confidence = result.get("confidence")

            if confidence is not None:
                label.config(
                    text=f"予測結果ですわ: {digit}（確信度: {confidence:.2%}）"
                )
            else:
                label.config(text=f"予測結果ですわ: {digit}")
        else:
            label.config(
                text=f"エラーですわね: {result.get('error', '原因不明のエラーですわ。。')}"
            )
    except Exception as e:
        label.config(text=f"接続エラーですわ: {e}")


predict_btn = tk.Button(root, text="判定しますわーーー", command=predict_digit)
predict_btn.pack(pady=5)

label = tk.Label(root, text="0〜9の数字を1つ描いてくださいましーーー")
label.pack(pady=5)

root.mainloop()