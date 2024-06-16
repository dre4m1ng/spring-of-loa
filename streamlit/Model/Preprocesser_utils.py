import numpy as np
import cv2
import os

class VideoPreprocessor:
    def __init__(self, video_path, save_folder, save_name_prefix, fps=10, save=False, size_type=1, quality=700):
        self.video_path = video_path
        self.save_folder = save_folder
        self.save_name_prefix = str(save_name_prefix)
        self.fps = fps
        self.size_type = size_type
        self.quality = quality
        self.save = save
        self.save_folder_cri = os.path.join(self.save_folder, "cri")
        self.save_folder_nor = os.path.join(self.save_folder, "nor")


    def preprocess_video(self):
        cap = cv2.VideoCapture(self.video_path)

        frame_interval = round(cap.get(cv2.CAP_PROP_FPS) / self.fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        if not os.path.exists(self.save_folder_cri):
            os.makedirs(self.save_folder_cri)
        
        if not os.path.exists(self.save_folder_nor):
            os.makedirs(self.save_folder_nor)

        while True:
            ret, frame = cap.read()


            if not ret:
                break

            if current_frame % frame_interval == 0:
                try:
                    preprocess_obj = preprocess(frame, self.save_folder, f"{self.save_name_prefix}_{current_frame}", save=self.save, size_type=self.size_type, quality=self.quality)    # ���⼭ ���忩�θ� ����

                    processed_yellow, processed_white = preprocess_obj.process_image()

                    out_path_cri = os.path.join(self.save_folder_cri, f"{self.save_name_prefix}_{current_frame}.jpg")
                    out_path_nor = os.path.join(self.save_folder_nor, f"{self.save_name_prefix}_{current_frame}.jpg")
                    cv2.imwrite(out_path_cri, processed_yellow)
                    cv2.imwrite(out_path_nor, processed_white)
                    print(f"{self.save_folder} 저장완료 ({current_frame}/{total_frames})")

                except Exception as e:
                    print(f"Error processing frame {current_frame}: {e}")

            current_frame += 1

        cap.release()
        print("작업완료")



class preprocess():
    def __init__(self, image_data, save_folder, save_name, save=False, size_type=1, quality=700):
        self.image = cv2.resize(image_data, (1920, 1080))
        self.save_folder = save_folder
        self.save_name = save_name
        self.save = save
        self.size_type = size_type
        self.quality = quality
        self.mask_regions = mask_regions = [
                                            ((0, 0), (888, 25)),
                                            ((705, 30), (1195, 120)),
                                            ((695, 165), (1110, 175)),
                                            ((0, 1052), (257, 1072)),
                                            ((650, 970), (1300, 1070)),
                                            ((430, 900), (870, 970)),
                                            ((1520, 1030), (1900, 1070)),
                                            ((0, 790), (430, 1020)),
                                            ((0, 340), (230, 640)),
                                            ((1630, 0), (1920, 300)),
                                            ((1570, 20), (1620, 150)),
                                            ((1630, 410), (1920, 650)),
                                            ((80, 50), (180, 95)),
                                            ((630, 100), (690, 130)),
                                            ((780, 30), (730, 45)),
                                        ]
        self.mask = self.create_mask(self.image.shape, self.mask_regions)



    def create_mask(self, image_shape, mask_regions):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)

        for region in mask_regions:
            cv2.rectangle(mask, region[0], region[1], 255, -1)

        return mask


    def apply_mask(self, save=False):
        mask_inv = cv2.bitwise_not(self.mask)

        masked_image = cv2.bitwise_and(self.image, self.image, mask=mask_inv)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_mask.jpg'), masked_image)

        self.image = masked_image

        return self.image



    def resize_image(self, image=None, save=False):
        if image is None:
            image = self.image

        if self.size_type == 1:
            image = cv2.resize(image, (self.quality, self.quality))
            image = cv2.resize(image, (1920, 1080))
        if self.size_type == 2:
            image = cv2.resize(image, (720, 405))
            image = cv2.resize(image, (1920, 1080))

        resize_image = image

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_quality.jpg'), resize_image)

        self.image = resize_image

        return self.image


    def extract_yellow(self, image=None, save=False):
        if image is None:
            image = self.image

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 160, 215])
        upper_yellow = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        result_image = image.copy()
        result_image[yellow_mask == 0] = [0, 0, 0]

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_extract_Y.jpg'), result_image)

        self.image = result_image

        return self.image


    def extract_white(self, image=None, save=False):
        if image is None:
            image = self.image

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])

        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        result_image = image.copy()
        result_image[white_mask == 0] = [0, 0, 0]

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_extract_W.jpg'), result_image)

        self.image = result_image
        return self.image


    def gaussian_blur(self, image=None, kernel_size=(5, 5), sigma_x=0, save=False):
        if image is None:
            image = self.image

        gaussian_image = cv2.GaussianBlur(image, kernel_size, sigma_x)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name+ '_gaussian.jpg'), gaussian_image)

        self.image = gaussian_image
        return self.image


    def median_blur(self, image=None, kernel_size=5, save=False):
        if image is None:
            image = self.image

        median_image = cv2.medianBlur(image, kernel_size)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_median.jpg'), median_image)

        self.image = median_image
        return self.image


    def binarize_image(self, image=None, save=False):
        if image is None:
            image = self.image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, otsu_thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_binary.jpg'), otsu_thresholded)

        self.image = otsu_thresholded
        return self.image


    def rgb_to_cmyk(self, image):

        rgb_image = image.astype(np.float32) / 255.0

        r = rgb_image[:,:,2]
        g = rgb_image[:,:,1]
        b = rgb_image[:,:,0]

        c = 1 - r
        m = 1 - g
        y = 1 - b

        k = np.minimum(c, np.minimum(m, y))

        c = (c - k) / (1 - k + 1e-10)
        m = (m - k) / (1 - k + 1e-10)
        y = (y - k) / (1 - k + 1e-10)

        k = k

        cmyk_image = np.stack([c, m, y, k], axis=-1) * 100
        cmyk_image = cmyk_image.astype(np.uint8)

        return cmyk_image


    def cmyk_transfer(self, image=None, save=False):
        if image is None:
            image = self.image

        ori_image = image

        cmyk_image = self.rgb_to_cmyk(ori_image)
        
        upper_bound_cmyk = np.array([0, 30, 100, 10])
        lower_bound_cmyk = np.array([0, 0, 60, 0])


        mask_cmyk = cv2.inRange(cmyk_image, lower_bound_cmyk, upper_bound_cmyk)

        result_cmyk = cv2.bitwise_and(ori_image, ori_image, mask=mask_cmyk)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_cymk.jpg'), result_cmyk)

        self.image = result_cmyk
        return self.image


    def process_image(self):

        self.resize_image()
        self.apply_mask()

        yellow_image = self.image.copy()
        white_image = self.image.copy()

        yellow_image = self.extract_yellow(yellow_image)
        yellow_binary = self.binarize_image(yellow_image)
        yellow_gaussian = self.gaussian_blur(yellow_binary)
        yellow_final = self.median_blur(yellow_gaussian)


        white_image = self.extract_white(white_image)
        white_binary = self.binarize_image(white_image)
        white_gaussian = self.gaussian_blur(white_binary)
        white_final = self.median_blur(white_gaussian)


        return yellow_final, white_final

