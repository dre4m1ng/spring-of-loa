import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
from multiprocessing import Pool

class Model:
    def __init__(self, image):
        self.image = image
        self.ocr = PaddleOCR(use_angle_cls=True, lang='korean', use_gpu=True)

    def extract_dmg(self, result_image):
        ocr_result = self.ocr.ocr(result_image, cls=True)
        
        if ocr_result and ocr_result[0] and ocr_result[0][0] and ocr_result[0][0][1]:
            text, confi = ocr_result[0][0][1]
            if isinstance(confi, float) and confi > 0.8:
                extracted_text = re.sub(r'[^0-9]', '', text)
                if extracted_text == '':
                    extracted_num = 0
                else:
                    extracted_num = int(extracted_text)
            else:
                extracted_num = 0
        else:
            extracted_num = 0
        return extracted_num
        
    def critical(self):
        lower_yellow = np.array([20, 160, 215])
        upper_yellow = np.array([40, 255, 255])

        cri_mask = cv2.inRange(self.image, lower_yellow, upper_yellow)

        cri_image = self.image.copy()
        cri_image[cri_mask == 0] = [0, 0, 0]
        return self.extract_dmg(cri_image)

    def normal(self):
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        
        nor_mask = cv2.inRange(self.image, lower_white, upper_white)
        
        nor_image = self.image.copy()
        nor_image[nor_mask == 0] = [0, 0, 0]
        return self.extract_dmg(nor_image)
    