import numpy as np

"""
drawing.py

Tools for frame annotation with alpha blending (looks better, less distracting, 
and does a better job of visually highlighting e.g. overlap regions, intersections,
etc.

ToDo:

circles? ellipses? text?

"""

def rect(frame, start_point, end_point,
         color, linewidth=1, alpha=0.5):
    """
    rect(frame, start_point, end_point, color, linewidth=1, alpha=0.5)

    Draw a rectangle with alpha blending

    :param frame: An image with shape (H,W,3)
    :param start_point: tuple (or array-like) of (x1, y1)
    :param end_point: tuple (or array-like) of (x2, y2)
    :param color: BGR 3-tuple
    :param linewidth: int, default=1
    :param alpha: float, default=0.5
    :return: frame
    """
    cvec = np.array(list(color)).reshape((1, 1, 3))

    x1, y1, x2, y2 = (start_point[0], start_point[1],
                      end_point[0], end_point[1])

    x1 = np.minimum(x1, frame.shape[1]-linewidth)
    x2 = np.minimum(x2, frame.shape[1]-linewidth)
    y1 = np.minimum(y1, frame.shape[0]-linewidth)
    y2 = np.minimum(y2, frame.shape[0]-linewidth)

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    wvec = np.ones(shape=(linewidth, w, 1))
    hvec = np.ones(shape=(h, linewidth, 1))

    warr = np.kron(wvec, cvec)
    harr = np.kron(hvec, cvec)

    x2 += 1
    y2 += 1

    frame[y1:(y1+linewidth), x1:x2, :] = alpha*warr + (1.-alpha)*frame[y1:(y1+linewidth), x1:x2, :]
    frame[(y2-linewidth):y2, x1:x2, :] = alpha*warr + (1.-alpha)*frame[(y2-linewidth):y2, x1:x2, :]

    frame[y1:y2, x1:(x1 + linewidth), :] = alpha * harr + (1. - alpha) * frame[y1:y2, x1:(x1 + linewidth), :]
    frame[y1:y2, (x2-linewidth):x2, :] = alpha * harr + (1. - alpha) * frame[y1:y2, (x2-linewidth):x2, :]

    return frame


def fill_rect(frame, start_point, end_point,
              color, alpha=0.5):
    """
    rect(frame, start_point, end_point, color, linewidth=1, alpha=0.5)

    :param frame: An image with shape (H,W,3)
    :param start_point: tuple (or array-like) of (x1, y1)
    :param end_point: tuple (or array-like) of (x2, y2)
    :param color: BGR 3-tuple
    :param alpha: float, default=0.5
    :return: frame
    """
    cvec = np.array(list(color)).reshape((1, 1, 3))

    x1, y1, x2, y2 = (start_point[0], start_point[1],
                      end_point[0], end_point[1])

    x1 = np.minimum(x1, frame.shape[1]-1)
    x2 = np.minimum(x2, frame.shape[1]-1)
    y1 = np.minimum(y1, frame.shape[0]-1)
    y2 = np.minimum(y2, frame.shape[0]-1)

    w = x2-x1+1
    h = y2-y1+1
    rectarr = np.ones(shape=(h, w, 1))
    arr = np.kron(rectarr, cvec)

    # print('Array shape = {}'.format(arr.shape))

    y2 += 1
    x2 += 1

    frame[y1:y2, x1:x2, :] = alpha*arr + (1.-alpha)*frame[y1:y2, x1:x2, :]

    return frame
