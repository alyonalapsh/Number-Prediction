import tkinter as tk
from PIL import Image, ImageDraw

import prediction


class Window:
    """
    Class to create a drawing window.

    Attributes:
        window: tkinter window instance.
        __canvas: tkinter canvas instance.
        __image: Image instance.
        __draw: ImageDraw instance.
        __line_points: List of points.
        __line_options: Dictionary of line options.

        """
    def __init__(self):
        """
        Initialize the window instance.
        """
        self.window = tk.Tk()
        self.__canvas = tk.Canvas(self.window, width=200, height=200)
        self.__canvas.pack()

        self.__image = Image.new('RGB', (200, 200), color='black')
        self.__draw = ImageDraw.Draw(self.__image)

        self.__line_points = []
        self.__line_options = {}

    def create_buttons(self):
        """
        Create the buttons in the window instance.
        """
        self.__canvas.bind('<Button-1>', self.__set_start)
        self.__canvas.bind('<B1-Motion>', self.__draw_line)
        self.__canvas.bind('<ButtonRelease-1>', self.__end_line)

        f_top = tk.Frame(self.window)
        f_bot = tk.Frame(self.window)

        button_pred = tk.Button(f_top, text="pred", command=self.__predict)
        button_close = tk.Button(f_bot, text="close", command=self.window.destroy)
        button_clear = tk.Button(f_top, text="clear", command=self.__clear)
        f_top.pack()
        f_bot.pack()

        button_pred.pack(side=tk.LEFT)
        button_close.pack(side=tk.LEFT)
        button_clear.pack(side=tk.LEFT)

    def __set_start(self, event):
        """
        Set the start of drawing.

        Parameters:
            event: set of coordinates.
        """
        self.__line_points.extend((event.x, event.y))

    def __draw_line(self, event):
        """
        Draw a line.

        Parameters:
            event: set of coordinates.
        """
        self.__line_points.extend((event.x, event.y))
        self.__canvas.create_line(self.__line_points, **self.__line_options, width=15)
        self.__draw.line(self.__line_points, **self.__line_options, width=15)

    def __end_line(self, event=None):
        """
        Locks the end of drawing.
        """
        self.__line_points.clear()

    def __predict(self):
        """
        Predict the class of drawing number.
        """
        pred = prediction.pred(self.__image)
        tk.Label(self.window, text=pred).pack(side=tk.TOP)

    def __clear(self):
        """
        Clears the drawing window and labels.
        """
        self.__canvas.delete('all')
        self.__draw.rectangle(xy=(0, 0, 200, 200), fill=(0, 0, 0))

        for child in self.window.winfo_children():
            if child.widgetName == "label":
                child.destroy()
