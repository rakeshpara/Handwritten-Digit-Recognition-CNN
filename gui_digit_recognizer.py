from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

# Load trained model
model = load_model('mnist.h5')

# -----------------------------------------------
# Enhanced preprocessing for accurate predictions
# -----------------------------------------------
def predict_digit(img):
    # Convert to grayscale and invert colors
    img = img.convert('L')
    img = np.array(img)
    img = 255 - img  # invert: white bg → black bg, black digit → white digit

    # Remove noise (threshold)
    img[img < 30] = 0

    # Crop to bounding box
    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1 + 1, x0:x1 + 1]
    else:
        img = np.zeros((28, 28))

    # Resize proportionally and center in 28x28 frame
    im = Image.fromarray(img)
    im.thumbnail((20, 20))
    new_img = Image.new('L', (28, 28), (0))
    new_img.paste(im, ((28 - im.size[0]) // 2, (28 - im.size[1]) // 2))

    # Normalize and reshape for model
    img = np.array(new_img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    preds = model.predict(img)
    preds = preds[0]  # flatten the 2D array [[...]] → [...]
    digit = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return digit, confidence


# -----------------------------------------------
# Tkinter GUI Application
# -----------------------------------------------
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        self.title("Handwritten Digit Recognizer")
        self.geometry("600x350")

        # Canvas and UI elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 32))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting, width=12, height=2)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all, width=12, height=2)

        # Layout
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=10)
        self.button_clear.grid(row=1, column=0, pady=10)

        # Bind drawing event
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.config(text="Draw a digit")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinates of the canvas
        a, b, c, d = rect
        rect = (a+8, b+8, c-8, d-8)  # tighter crop for better accuracy

        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.config(text=f"{digit}, {int(acc*100)}%")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 6  # smaller radius for thinner strokes
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='black')

# Run the app
app = App()
mainloop()
