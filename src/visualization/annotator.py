from random import randint
import cv2

from src.visualization.drawing import *


class Annotator:

    def __init__(self):
        pass

    @classmethod
    def random_color(cls):
        return (randint(0,255), randint(0,255), randint(0,255))
    
    def annotate(self, frame, bbox:tuple, label:str = '', **kwargs):
        pass


class BoxAnnotator(Annotator):

    def __init__(self, 
                 fill_bbox=True,
                 outline_bbox=True,
                 fill_alpha=0.1,
                 outline_alpha=0.66,
                 outline_color=None,
                 fill_color=None,
                 outline_width=1):

        self.fill_bbox = fill_bbox
        self.outline_bbox = outline_bbox
        self.fill_alpha = fill_alpha
        self.outline_alpha = outline_alpha
        if outline_color is None:
            outline_color = Annotator.random_color()
        if fill_color is None:
            fill_color = outline_color

        self.outline_color = outline_color
        self.fill_color = fill_color

        self.outline_width = outline_width        

    def annotate(self, frame, bbox, label='', **kwargs):
           
        x1,y1,x2,y2 = bbox
        if self.fill_bbox:
            fill_rect(frame, (x1, y1), (x2, y2), color=self.fill_color, alpha=self.fill_alpha)
        if self.outline_bbox:
            rect(frame, (x1, y1), (x2, y2),
                 linewidth=1, color=self.outline_color, alpha=self.outline_alpha)
        return frame


class TextAnnotator(Annotator):

    def __init__(self, 
                 text_color=None,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 text_size=0.5,
                 text_width=1, 
                 vert_offset=15):

        if text_color is None:
            text_color = Annotator.random_color()

        self.text_color = text_color
        self.font = font
        self.text_size = text_size
        self.text_width = text_width
        self.vert_offset = vert_offset

    def annotate(self, frame, bbox, label='', **kwargs):
        x1,y1,x2,y2 = bbox
        v = self.vert_offset
        y_text = y1 - v if y1 - v > v else y1 + v
        cv2.putText(frame, label, (x1, y_text),
                    self.font, self.text_size, 
                    self.text_color, self.text_width)

