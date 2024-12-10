import tkinter as tk
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import Prediction


line_id = None
line_points = []
line_options = {}

window = tk.Tk()
canvas = tk.Canvas(window, width=200, height=200)
canvas.pack()

my_image = Image.new('RGB', (200, 200), color='black')
draw = ImageDraw.Draw(my_image)


def draw_line(event):
    global line_id
    line_points.extend((event.x, event.y))

    line_id = canvas.create_line(line_points, **line_options, width=20)
    draw.line(line_points, **line_options, width=20)


def set_start(event):
    line_points.extend((event.x, event.y))


def end_line(event=None):
    global line_id
    line_points.clear()
    line_id = None


def predict():
    sample = Prediction.transform(my_image)
    plt.imshow(sample.view(28, 28, 1), cmap='gray')
    plt.show()

    pred = Prediction.pred(my_image)
    tk.Label(window, text=pred).pack(side=tk.TOP)


def clear():
    canvas.delete('all')
    draw.rectangle(xy=(0, 0, 200, 200), fill=(0, 0, 0))

    sample = Prediction.transform(my_image)
    plt.imshow(sample.view(28, 28, 1), cmap='gray')
    plt.show()

    for child in window.winfo_children():
        if child.widgetName == "label":
            child.destroy()


def create_buttons():
    canvas.bind('<Button-1>', set_start)
    canvas.bind('<B1-Motion>', draw_line)
    canvas.bind('<ButtonRelease-1>', end_line)

    f_top = tk.Frame(window)
    f_bot = tk.Frame(window)

    button_pred = tk.Button(f_top, text="pred", command=predict)
    button_close = tk.Button(f_bot, text="close", command=window.destroy)
    button_clear = tk.Button(f_top, text="clear", command=clear)

    f_top.pack()
    f_bot.pack()

    button_pred.pack(side=tk.LEFT)
    button_close.pack(side=tk.LEFT)
    button_clear.pack(side=tk.LEFT)
