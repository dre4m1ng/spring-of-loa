import re
import json
import os
import glob
import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

class OCRModel:
    def __init__(self, lang='korean', use_gpu=True):
        """
        OCRModel 클래스를 초기화합니다.

        Args:
            lang (str, optional): OCR 모델의 언어. 기본값은 'korean'입니다.
            use_gpu (bool, optional): GPU 사용 여부. 기본값은 True입니다.
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

    def process_images(self, yellow_folder_path, white_folder_path, confidence_threshold=0.8):
        """
        노란색 폴더와 하얀색 폴더의 이미지에서 텍스트를 추출하고 결과를 반환합니다.

        Args:
            yellow_folder_path (str): 노란색 이미지가 있는 폴더 경로
            white_folder_path (str): 하얀색 이미지가 있는 폴더 경로
            confidence_threshold (float, optional): 텍스트 인식 신뢰도 임계값. 기본값은 0.8입니다.

        Returns:
            list: 각 이미지에서 추출된 숫자 정보가 포함된 딕셔너리 리스트
        """
        results = []
        all_critical_numbers = set()
        all_normal_numbers = set()

        # 노란색 이미지 처리
        self._process_folder(yellow_folder_path, results, all_critical_numbers, all_normal_numbers, is_yellow=True, confidence_threshold=confidence_threshold)

        # 하얀색 이미지 처리
        self._process_folder(white_folder_path, results, all_critical_numbers, all_normal_numbers, is_yellow=False, confidence_threshold=confidence_threshold)

        # Critical과 Normal이 모두 빈 리스트인 경우 제외
        filtered_results = [result for result in results if result['Critical'] or result['Normal']]

        return filtered_results

    def _process_folder(self, folder_path, results, all_critical_numbers, all_normal_numbers, is_yellow, confidence_threshold):
        """
        폴더의 이미지에서 텍스트를 추출하고 결과를 저장합니다.

        Args:
            folder_path (str): 이미지 폴더 경로
            results (list): 결과를 저장할 리스트
            all_critical_numbers (set): 모든 Critical 값을 저장할 set
            all_normal_numbers (set): 모든 Normal 값을 저장할 set
            is_yellow (bool): 노란색 이미지인지 여부
            confidence_threshold (float): 텍스트 인식 신뢰도 임계값
        """
        image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        image_paths.sort(key=self.extract_number)

        for image_path in image_paths:
            try:
                # 이미지 로드
                image = cv2.imread(image_path)

                # 이미지에서 텍스트 추출
                result = self.ocr.ocr(image, cls=True)

                # 이미지 파일 이름 추출
                image_name = os.path.basename(image_path)

                critical_numbers = []
                normal_numbers = []
                ocr_results = [] # OCR 결과를 저장할 리스트

                if result is not None and len(result) > 0:
                    for line in result:
                        for detection in line:
                            box, text_info = detection
                            text, confidence = text_info
                            if isinstance(confidence, float) and confidence > confidence_threshold:
                                extracted_text = re.sub(r'[^0-9]', '', text)
                                if len(extracted_text) >= 5:
                                    if is_yellow and int(extracted_text) not in all_critical_numbers:
                                        critical_numbers.append(int(extracted_text))
                                        all_critical_numbers.add(int(extracted_text))
                                    elif not is_yellow and int(extracted_text) not in all_normal_numbers:
                                        normal_numbers.append(int(extracted_text))
                                        all_normal_numbers.add(int(extracted_text))
                                ocr_results.append((text, confidence, box)) # OCR 결과 저장

                if is_yellow:
                    results.append({"Image Name": image_name, "Critical": critical_numbers, "Normal": [], "OCR_Results": ocr_results})
                else:
                    result_item = next((item for item in results if item["Image Name"] == image_name), None)
                    if result_item:
                        result_item["Normal"] = normal_numbers
                        result_item["OCR_Results"] = ocr_results

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

    @staticmethod
    def extract_number(filename):
        """
        파일 이름에서 숫자를 추출합니다.

        Args:
            filename (str): 파일 이름

        Returns:
            int: 추출된 마지막 숫자, 숫자가 없으면 -1 반환
        """
        s = re.findall(r'\d+', filename)
        return int(s[-1]) if s else -1

    def save_results(self, results, output_path):
        """
        결과를 JSON 파일로 저장합니다.

        Args:
            results (list): 각 이미지에서 추출된 숫자 정보가 포함된 딕셔너리 리스트
            output_path (str): 출력 파일 경로
        """
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)
        except (IOError, OSError) as e:
            print(f"Error writing results to {output_path}: {e}")
    

class Postprocessing:   ##### 후처리를 정해서 골라야함.
    def preprocess_data(self, data_list):   ##### 모델의 간단히 처리하는게 있는데 그럼 시간 2배.
        """
        입력 데이터를 정수로 변환하는 메소드
        
        Args:
            data_list: 처리할 데이터 리스트
        
        Returns:
            정수로 변환된 데이터 리스트
        """
        preprocessed = []
        for data in data_list:
            if isinstance(data, str):
                # 문자열에서 숫자가 아닌 문자 제거
                cleaned_value = re.sub(r'[^0-9]', '', data)
                if cleaned_value:
                    preprocessed.append(int(cleaned_value))
            elif isinstance(data, (int, float)):
                preprocessed.append(int(data))
        return preprocessed

    def remove_outliers_iqr(self, data, multiplier=5):
        """
        IQR 방법을 사용하여 이상치를 제거하는 메소드
        
        Args:
            data: 처리할 데이터 리스트
            multiplier: IQR에 곱할 값 (기본값: 5)
        
        Returns:
            이상치가 제거된 데이터 리스트와 제거된 이상치 리스트
        """
        if not data:
            return [], []
        data = np.array(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + multiplier * iqr
        filtered_data = data[data <= upper_bound].tolist()
        removed_outliers = data[data > upper_bound].tolist()
        return filtered_data, removed_outliers

    def clean_ocr_results_iqr(self, results, gt_path, multiplier=5):
        """
        OCR 결과를 정제하고 ground truth와 비교하여 이상치를 제거하는 메소드
        
        Args:
            results: OCR 결과 리스트
            gt_path: ground truth 데이터 파일 경로
            multiplier: IQR에 곱할 값 (기본값: 5)
        
        Returns:
            정제된 OCR 결과와 제거된 이상치 정보
        """
        # Ground truth 데이터 로드
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except (IOError, OSError) as e:
            print(f"Error reading ground truth data from {gt_path}: {e}")
            return None, None, None

        # 모든 Critical과 Normal 데이터 수집
        all_critical = []
        all_normal = []
        for result in results:
            if 'Critical' in result:
                all_critical.extend(self.preprocess_data(result['Critical']))
            if 'Normal' in result:
                all_normal.extend(self.preprocess_data(result['Normal']))

        # IQR 방법으로 이상치 제거
        filtered_critical, removed_critical = self.remove_outliers_iqr(all_critical, multiplier)
        filtered_normal, removed_normal = self.remove_outliers_iqr(all_normal, multiplier)

        # 정제된 결과 생성
        clean_results = []
        for result in results:
            image_name = result['Image Name']
            result['Critical'] = [value for value in self.preprocess_data(result['Critical']) if value in filtered_critical]
            result['Normal'] = [value for value in self.preprocess_data(result['Normal']) if value in filtered_normal]
            clean_results.append({"Image Name": image_name, "Critical": result['Critical'], "Normal": result['Normal']})

        # Ground truth에서 잘못 제거된 값 추적
        removed_critical_from_gt = []
        removed_normal_from_gt = []
        removed_critical_image_names = []
        removed_normal_image_names = []

        for gt_item in gt_data:
            gt_critical = set(gt_item['Critical'])
            gt_normal = set(gt_item['Normal'])

            wrongly_removed_critical = gt_critical.intersection(removed_critical)
            wrongly_removed_normal = gt_normal.intersection(removed_normal)

            if wrongly_removed_critical:
                removed_critical_from_gt.extend(wrongly_removed_critical)
                removed_critical_image_names.append(gt_item['Image Name'])

            if wrongly_removed_normal:
                removed_normal_from_gt.extend(wrongly_removed_normal)
                removed_normal_image_names.append(gt_item['Image Name'])

        return clean_results, removed_critical, removed_normal, removed_critical_from_gt, removed_normal_from_gt, removed_critical_image_names, removed_normal_image_names

    def differ_by_one_or_two_digits(self, num1, num2, allowed_diff=1):
        """
        두 숫자가 지정된 자릿수 이하로 다른지 확인하는 메소드
        
        Args:
            num1, num2: 비교할 두 숫자
            allowed_diff: 허용되는 다른 자릿수 (기본값: 1)
        
        Returns:
            두 숫자가 허용된 자릿수 이하로 다르면 True, 아니면 False
        """
        str1, str2 = str(num1), str(num2)
        if len(str1) != len(str2):
            return False
        diff_count = sum(1 for a, b in zip(str1, str2) if a != b)
        return diff_count <= allowed_diff

    def find_groups(self, numbers_list, allowed_diff=1):
        """
        비슷한 숫자들을 그룹화하는 메소드
        
        Args:
            numbers_list: 그룹화할 숫자 리스트
            allowed_diff: 허용되는 다른 자릿수 (기본값: 1)
        
        Returns:
            그룹화된 숫자 집합들의 리스트
        """
        groups = []
        visited = set()
        for num in numbers_list:
            if num not in visited:
                group = {num}
                for other in numbers_list:
                    if other != num and self.differ_by_one_or_two_digits(num, other, allowed_diff):
                        group.add(other)
                        visited.add(other)
                groups.append(group)
                visited.update(group)
        return groups

    def keep_extreme_numbers(self, groups, keep_largest=True):
        """
        각 그룹에서 극단값(최대값 또는 최소값)을 선택하는 메소드
        
        Args:
            groups: 숫자 그룹들의 리스트
            keep_largest: True면 최대값, False면 최소값 선택 (기본값: True)
        
        Returns:
            선택된 극단값들의 집합
        """
        extreme_numbers = set()
        for group in groups:
            if keep_largest:
                extreme_numbers.add(max(group))
            else:
                extreme_numbers.add(min(group))
        return extreme_numbers

    def remove_similar_numbers(self, cleaned_results):
        """
        비슷한 숫자들 중에서 극단값만 유지하고 나머지는 제거하는 메소드
        
        Args:
            cleaned_results: 정제된 OCR 결과 리스트
        
        Returns:
            극단값만 유지된 결과와 제거된 숫자들
        """
        # 모든 Critical과 Normal 데이터 수집
        all_critical = [value for result in cleaned_results for value in result['Critical']]
        all_normal = [value for result in cleaned_results for value in result['Normal']]

        # 비슷한 숫자 그룹화
        critical_groups = self.find_groups(all_critical, allowed_diff=2)
        normal_groups = self.find_groups(all_normal, allowed_diff=2)

        # 각 그룹에서 최대값만 유지
        kept_critical_numbers = self.keep_extreme_numbers(critical_groups, keep_largest=True)
        kept_normal_numbers = self.keep_extreme_numbers(normal_groups, keep_largest=True)

        # 제거된 숫자 추적
        removed_critical_numbers = set(all_critical) - kept_critical_numbers
        removed_normal_numbers = set(all_normal) - kept_normal_numbers

        # 결과 업데이트
        for result in cleaned_results:
            result['Critical'] = [value for value in result['Critical'] if value in kept_critical_numbers]
            result['Normal'] = [value for value in result['Normal'] if value in kept_normal_numbers]

        return cleaned_results, removed_critical_numbers, removed_normal_numbers