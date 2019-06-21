import numpy as np
from PIL import Image, ImageDraw

def draw_bbox(img, bbox, text):
    h, w = img.shape[0], img.shape[1]
    x0, y0, x1, y1 = max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], w-1), min(bbox[3], h-1)
    drawed_img = img * 1
    color = np.zeros([x1 - x0 + 1, 3])
    color[:, 1] = np.ones([x1 - x0 + 1]) * 255#Green rectangle
    drawed_img[y0, x0:x1 + 1, :] = color
    drawed_img[y1, x0:x1 + 1, :] = color
    color = np.zeros([y1 - y0 + 1, 3])
    color[:, 1] = np.ones([y1 - y0 + 1]) * 255  # Green rectangle
    drawed_img[y0:y1 + 1, x0, :] = color
    drawed_img[y0:y1 + 1, x1, :] = color
    #type text
    drawed_img = Image.fromarray(np.uint8(drawed_img))
    draw = ImageDraw.Draw(drawed_img)
    x = int(bbox[0])
    y = int(bbox[1])
    draw.text((x, y), text)
    drawed_img = np.array(drawed_img)
    return drawed_img