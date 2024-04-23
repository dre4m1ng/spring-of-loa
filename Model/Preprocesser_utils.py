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

    def preprocess_video(self, operations):
        # cv2로 영상 파일 읽기
        cap = cv2.VideoCapture(self.video_path)

        # 설정한 FPS에 맞춰 간격을 계산
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / self.fps)  # fps = 내가 원하는 output의 fps 10 fps = 60/10 => 1초에 10장이 나오는 6fps으로 나눈것.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        # 저장 폴더가 없다면 생성
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        while True:
            # 영상에서 프레임을 하나씩 읽어옴
            ret, frame = cap.read()


            # 더 이상 읽을 프레임이 없으면 종료
            if not ret:
                break

            # 설정한 FPS에 맞춰 프레임 저장
            if current_frame % frame_interval == 0:
                try:
                    # 전처리 객체 생성 및 전처리 수행
                    preprocess_obj = preprocess(frame, self.save_folder, f"{self.save_name_prefix}_{current_frame}", save=self.save, size_type=self.size_type, quality=self.quality)    # 여기서 저장여부를 결정

                    processed_frame = preprocess_obj.process_image(operations)

                    # 전처리된 프레임을 다시 영상으로 저장
                    out_path = os.path.join(self.save_folder, f"{self.save_name_prefix}_{current_frame}.jpg")
                    cv2.imwrite(out_path, processed_frame)
                    print(f"{out_path} 저장 완료 ({current_frame}/{total_frames})")

                except Exception as e:
                    print(f"Error processing frame {current_frame}: {e}")

            # 프레임 번호 업데이트
            current_frame += 1

        # 자원 해제
        cap.release()
        print("작업 완료")





class preprocess():
    def __init__(self, image_data, save_folder, save_name, save=False, size_type=1, quality=700):
        self.image = cv2.resize(image_data, (1920, 1080))
        self.save_folder = save_folder
        self.save_name = save_name
        self.save = save
        self.size_type = size_type
        self.quality = quality
        self.mask_regions = mask_regions = [
                                            ((0, 0), (888, 25)),  # 상단 상태 표시줄
                                            ((705, 30), (1195, 120)), # 보스 체력, 이름 게이지
                                            ((695, 165), (1110, 175)), # 보스 상태이상
                                            ((0, 1052), (257, 1072)), # 하단 레벨 표시
                                            ((650, 970), (1300, 1070)), # 스킬, 체력 게이지
                                            ((430, 900), (870, 970)), # 버프 표시줄
                                            ((1520, 1030), (1900, 1070)), # 설정창
                                            ((0, 790), (430, 1020)), # 채팅창
                                            ((0, 340), (230, 640)), # 파티창
                                            ((1630, 0), (1920, 300)), # 미니맵
                                            ((1570, 20), (1620, 150)), # 미니맵 옆
                                            ((1630, 410), (1920, 650)), # 퀘스트창
                                            ((80, 50), (180, 95)), # 레이드 이름
                                            ((630, 100), (690, 130)), # 광폭화까지 타이머
                                            ((780, 30), (730, 45)), # 보스
                                        ]
        self.mask = self.create_mask(self.image.shape, self.mask_regions)


    def create_mask(self, image_shape, mask_regions):
        # 마스크 이미지 생성
        mask = np.zeros(image_shape[:2], dtype=np.uint8)

        # 마스크 영역을 흰색으로 채우기
        for region in mask_regions:
            cv2.rectangle(mask, region[0], region[1], 255, -1)
        return mask


    def apply_mask(self, save=False):
        # 마스크 반전
        mask_inv = cv2.bitwise_not(self.mask)

        # 적용
        masked_image = cv2.bitwise_and(self.image, self.image, mask=mask_inv)

        # 최종 결과를 저장
        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_mask.jpg'), masked_image)

        self.image = masked_image
        return self.image


    def resize_image(self, image=None, save=False):   # 해상도 조절
        if image is None:
            image = self.image

        if self.size_type == 1:
            # 이미지를 700x700으로 줄임
            image = cv2.resize(image, (self.quality, self.quality))   # 700, 750, 800, 850 1:1 비율
            # 다시 1920x1080으로 확대
            image = cv2.resize(image, (1920, 1080))
        if self.size_type == 2:
            # 이미지를 720x405 줄임
            image = cv2.resize(image, (720, 405))   # 720:405 비율
            # 다시 1920x1080으로 확대
            image = cv2.resize(image, (1920, 1080))

        resize_image = image
        # 최종 결과를 저장
        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_quality.jpg'), resize_image)

        self.image = resize_image
        return self.image


    def extract_yellow(self, image=None, save=False):   # 이미지 처리
        if image is None:
            image = self.image

        # HSV(Hue, Saturation, Value)로 이미지를 변환합니다.
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 노란색의 HSV 범위를 정의합니다.                               # 남궁맑음
        lower_yellow = np.array([20, 160, 215])  # 노란색의 하한값      # [30, 150, 120]
        upper_yellow = np.array([40, 255, 255])  # 노란색의 상한값      # [90, 255, 255]

        # HSV 이미지에서 노란색 범위에 해당하는 픽셀을 찾습니다.
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # 노란색을 제외한 나머지 부분을 검은색으로 대체합니다.
        # 새 이미지 생성
        result_image = image.copy()
        result_image[yellow_mask == 0] = [0, 0, 0]

        # 최종 결과물 저장
        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_extract_Cri.jpg'), result_image)

        self.image = result_image
        return self.image


    def extract_white(self, image=None, save=False):
        if image is None:
            image = self.image

        # HSV(Hue, Saturation, Value)로 이미지를 변환합니다.
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 하얀색의 HSV 범위를 정의합니다. (하얀색은 채도가 낮고 명도가 높습니다.)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])

        # HSV 이미지에서 하얀색 범위에 해당하는 픽셀을 찾습니다.
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # 하얀색에 해당하는 마스크를 사용하여 해당 색상만 추출합니다.
        # 새 이미지 생성
        result_image = image.copy()
        result_image[white_mask == 0] = [0, 0, 0]

        # 최종 결과물 저장
        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_extract_Norm.jpg'), result_image)

        self.image = result_image
        return self.image


    def gaussian_blur(self, image=None, kernel_size=(5, 5), sigma_x=0, save=False):
        """
        이미지에 가우시안 블러를 적용하는 함수
        :param image: 원본 이미지
        :param kernel_size: 커널 크기, (너비, 높이) 형태의 튜플. 커널 크기가 커질수록 더 많이 흐려짐.
        :param sigma_x: 가우시안 커널의 X 방향 표준편차. 0이면 커널 크기로부터 자동 계산됨.
        :return: 가우시안 블러가 적용된 이미지
        """
        if image is None:
            image = self.image

        gaussian_image = cv2.GaussianBlur(image, kernel_size, sigma_x)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name+ '_gaussian.jpg'), gaussian_image)

        self.image = gaussian_image
        return self.image


    def median_blur(self, image=None, kernel_size=5, save=False):
        """
        지정된 이미지에 미디언 블러를 적용하는 함수입니다.
        :param image: 처리할 이미지의 파일 경로입니다.
        :param kernel_size: 미디언 블러를 적용할 때 사용할 커널 크기입니다. 기본값은 5입니다.
        :param show_result: 처리된 이미지를 보여줄지 여부입니다. 기본값은 True입니다.
        """
        if image is None:
            image = self.image

        # 미디언 블러를 적용합니다.
        median_image = cv2.medianBlur(image, kernel_size)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_median.jpg'), median_image)

        self.image = median_image
        return self.image


    def binarize_image(self, image=None, save=False):
        if image is None:
            image = self.image

        # 이미지를 그레이스케일로 로드
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 이진화 수행
        _, otsu_thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_binary.jpg'), otsu_thresholded)

        self.image = otsu_thresholded
        return self.image


    # RGB to CMYK 변환 함수 -> 폐기 ....가능성 다분
    def rgb_to_cmyk(self, image):

        # 이미지를 0~1 사이로 정규화
        rgb_image = image.astype(np.float32) / 255.0

        r = rgb_image[:,:,2]
        g = rgb_image[:,:,1]
        b = rgb_image[:,:,0]

        c = 1 - r
        m = 1 - g
        y = 1 - b

        k = np.minimum(c, np.minimum(m, y))

        c = (c - k) / (1 - k + 1e-10)  # 0으로 나누는 오류를 피하기 위해 작은 값 추가
        m = (m - k) / (1 - k + 1e-10)
        y = (y - k) / (1 - k + 1e-10)

        # K 값은 그대로 사용
        k = k

        # CMYK 이미지 생성 (0~100% 범위로 변환)
        cmyk_image = np.stack([c, m, y, k], axis=-1) * 100
        cmyk_image = cmyk_image.astype(np.uint8)
        return cmyk_image


    def cmyk_transfer(self, image=None, save=False):
        if image is None:
            image = self.image

        # 오리지널 이미지
        ori_image = image

        # RGB 이미지를 CMYK로 변환
        cmyk_image = self.rgb_to_cmyk(ori_image)

        # CMYK 색상 범위 설정 (0~100%)
                                    # [C, M, Y, K]
        # upper_bound_cmyk = np.array([100, 100, 100, 100])  # 상한값
        # lower_bound_cmyk = np.array([0, 0, 0, 0])  # 하한값
        upper_bound_cmyk = np.array([0, 30, 100, 10])  # 상한값
        lower_bound_cmyk = np.array([0, 0, 60, 0])  # 하한값


        # 특정 색상 범위에 해당하는 부분을 이진화하여 마스크 생성
        mask_cmyk = cv2.inRange(cmyk_image, lower_bound_cmyk, upper_bound_cmyk)

        # 특정 색상 부분만 추출
        result_cmyk = cv2.bitwise_and(ori_image, ori_image, mask=mask_cmyk)

        if self.save:
            cv2.imwrite(os.path.join(self.save_folder, self.save_name + '_cymk.jpg'), result_cmyk)

        self.image = result_cmyk
        return self.image


    def process_image(self, operations):
        """
        전처리 테스트용 함수.
        operations: 실행하고자 하는 작업들의 리스트. 예: ['apply_mask', 'resize_image', 'extract_yellow', 'gaussian_blur']
        """
        image = self.image
        for operation in operations:
            if hasattr(self, operation):
                method = getattr(self, operation)
                image = method(image)
            else:
                print(f"{operation} 메서드는 존재하지 않습니다.")

        return image