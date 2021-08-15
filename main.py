import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'left': 160, 'top': 160, 'width': 200, 'height': 200}

with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGB',
            (screenShot.width, screenShot.height),
            screenShot.rgb,
        )
        cv2.imshow('test', np.array(img))
        if cv2.waitKey(33) & 0xFF in (
            ord('q'),
            27,
        ):
            break