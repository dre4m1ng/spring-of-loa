import re
import json
import os
import glob
import cv2
import paddle
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import wandb

class OCRModel_DB:
    def __init__(self, lang='korean', use_gpu=True, project_name="ocr", entity="sol_of_loa", run_name=None):
        """
        OCRModel 클래스를 초기화합니다.

        Args:
            lang (str, optional): OCR 모델의 언어. 기본값은 'korean'입니다.
            use_gpu (bool, optional): GPU 사용 여부. 기본값은 True입니다.
            project_name (str, optional): wandb 프로젝트 이름. 기본값은 'OCR'입니다.
            run_name (str, optional): wandb 실행 이름. 기본값은 None입니다.
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

        # wanDB init
        wandb.init(project=project_name, entity="sol_of_loa", name=run_name)

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

    def evaluate(self, results, gt_path):
        """
        결과를 Ground Truth 데이터와 비교하여 정밀도, 재현율, 정확도를 계산합니다.

        Args:
            results (list): 각 이미지에서 추출된 숫자 정보가 포함된 딕셔너리 리스트
            gt_path (str): Ground Truth 데이터 파일 경로

        Returns:
            tuple: Critical과 Normal에 대한 정밀도, 재현율, 정확도
        """
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except (IOError, OSError) as e:
            print(f"Error reading ground truth data from {gt_path}: {e}")
            return None
        
        # results 인자의 타입 확인
        if isinstance(results, str):
            # JSON 파일 경로인 경우 파일에서 읽어오기
            try:
                with open(results, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f"Error reading results from {results}: {e}")
                return None
        
        elif isinstance(results, list):
            # 리스트인 경우 그대로 사용
            pass
        else:
            print("Invalid results format. Expected JSON file path or list.")
            return None

        total_critical_numbers = 0
        total_normal_numbers = 0
        correct_critical_count = 0
        correct_normal_count = 0
        false_critical_count = 0
        false_normal_count = 0
        false_negative_critical_count = 0
        false_negative_normal_count = 0

        for i, gt_item in enumerate(gt_data):
            gt_critical = gt_item['Critical']
            gt_normal = gt_item['Normal']

            total_critical_numbers += len(gt_critical)
            total_normal_numbers += len(gt_normal)

            result_item = next((item for item in results if item['Image Name'] == gt_item['Image Name']), None)

            if result_item is not None:
                pred_critical = result_item['Critical']
                pred_normal = result_item['Normal']

                # Critical 숫자 비교
                correct_critical_count += sum(1 for x in pred_critical if x in gt_critical)
                false_critical_count += sum(1 for x in pred_critical if x not in gt_critical)
                false_negative_critical_count += sum(1 for x in gt_critical if x not in pred_critical)

                # Normal 숫자 비교
                correct_normal_count += sum(1 for x in pred_normal if x in gt_normal)
                false_normal_count += sum(1 for x in pred_normal if x not in gt_normal)
                false_negative_normal_count += sum(1 for x in gt_normal if x not in pred_normal)
            else:
                false_critical_count += len(gt_critical)
                false_normal_count += len(gt_normal)
                false_negative_critical_count += len(gt_critical)
                false_negative_normal_count += len(gt_normal)
            
            # wandb에 변수 값 로깅
            wandb.log({
                "correct_critical_count": correct_critical_count,
                "correct_normal_count": correct_normal_count,
                "false_critical_count": false_critical_count,
                "false_normal_count": false_normal_count,
                "false_negative_critical_count": false_negative_critical_count,
                "false_negative_normal_count": false_negative_normal_count
            }, step=i)

        critical_precision = correct_critical_count / (correct_critical_count + false_critical_count) if (correct_critical_count + false_critical_count) > 0 else 0
        normal_precision = correct_normal_count / (correct_normal_count + false_normal_count) if (correct_normal_count + false_normal_count) > 0 else 0

        critical_recall = correct_critical_count / (correct_critical_count + false_negative_critical_count) if (correct_critical_count + false_negative_critical_count) > 0 else 0
        normal_recall = correct_normal_count / (correct_normal_count + false_negative_normal_count) if (correct_normal_count + false_negative_normal_count) > 0 else 0

        critical_accuracy = correct_critical_count / total_critical_numbers if total_critical_numbers > 0 else 0
        normal_accuracy = correct_normal_count / total_normal_numbers if total_normal_numbers > 0 else 0

        # 전체 정확도 계산
        total_correct_count = correct_critical_count + correct_normal_count
        total_numbers = total_critical_numbers + total_normal_numbers
        total_accuracy = total_correct_count / total_numbers if total_numbers > 0 else 0

        # wandb에 메트릭 로깅
        wandb.log({
            "critical_precision": critical_precision,
            "critical_recall": critical_recall,
            "critical_accuracy": critical_accuracy,
            "normal_precision": normal_precision,
            "normal_recall": normal_recall,
            "normal_accuracy": normal_accuracy,
            "total_accuracy": total_accuracy
        })

        return critical_precision, critical_recall, critical_accuracy, normal_precision, normal_recall, normal_accuracy, total_accuracy
    
    def calculate_total_damage(self, results):
        """
        결과에서 Critical과 Normal 데미지의 합계를 계산합니다.

        Args:
            results (list): 각 이미지에서 추출된 숫자 정보가 포함된 딕셔너리 리스트

        Returns:
            tuple: Critical 데미지 합계, Normal 데미지 합계, 전체 데미지 합계
        """
        total_critical_damage = sum(sum(result_item['Critical']) for result_item in results)
        total_normal_damage = sum(sum(result_item['Normal']) for result_item in results)

        total_damage = total_critical_damage + total_normal_damage

        return total_critical_damage, total_normal_damage, total_damage
    
    def draw_text(self, image, text, position, color):
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    def visualize_errors(self, results, gt_path, yellow_folder_path, white_folder_path, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except (IOError, OSError) as e:
            print(f"Error reading ground truth data from {gt_path}: {e}")
            return None

        # results 인자의 타입 확인
        if isinstance(results, str):
            # JSON 파일 경로인 경우 파일에서 읽어오기
            try:
                with open(results, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f"Error reading results from {results}: {e}")
                return None
        
        elif isinstance(results, list):
            # 리스트인 경우 그대로 사용
            pass
        else:
            print("Invalid results format. Expected JSON file path or list.")
            return None

        total_images = len(results)
        predicted_critical_count = 0
        predicted_normal_count = 0
        false_critical_count = 0
        false_normal_count = 0
        no_gt_critical_count = 0
        no_gt_normal_count = 0
        saved_image_count = 0
        model_performance = 0

        for result_item in results:
            image_name = result_item['Image Name']
            image_modified = False

            # Ground truth 데이터에서 해당 이미지의 정보 가져오기
            gt_item = next((item for item in gt_data if item["Image Name"] == image_name), None)

            if gt_item is None:
                print(f"No ground truth data found for image: {image_name}")
                
                if result_item['Critical']:
                    no_gt_critical_count += len(result_item['Critical'])
                    image_path = os.path.join(yellow_folder_path, image_name)
                elif result_item['Normal']:
                    no_gt_normal_count += len(result_item['Normal'])
                    image_path = os.path.join(white_folder_path, image_name)
                else:
                    continue
                
                image = cv2.imread(image_path)
                if image is not None:
                    text_position = (10, 30)
                    
                    if result_item['Critical']:
                        self.draw_text(image, f"Predicted Critical: {', '.join(map(str, result_item['Critical']))}", text_position, (0, 0, 255))
                        text_position = (text_position[0], text_position[1] + 30)
                    
                    if result_item['Normal']:
                        self.draw_text(image, f"Predicted Normal: {', '.join(map(str, result_item['Normal']))}", text_position, (255, 0, 0))
                    
                    image_modified = True

            else:
                gt_critical = set(gt_item['Critical'])
                gt_normal = set(gt_item['Normal'])

                pred_critical = set(result_item['Critical'])
                pred_normal = set(result_item['Normal'])
                text_position = (10, 30)

                false_critical = pred_critical - gt_critical
                false_normal = pred_normal - gt_normal

                if false_critical:
                    false_critical_count += len(false_critical)
                    image_path = os.path.join(yellow_folder_path, image_name)
                elif false_normal:
                    false_normal_count += len(false_normal)
                    image_path = os.path.join(white_folder_path, image_name)
                else:
                    continue

                image = cv2.imread(image_path)
                if image is not None:
                    if false_critical:
                        self.draw_text(image, f"False Critical: {', '.join(map(str, false_critical))}", text_position, (0, 0, 255))
                        text_position = (text_position[0], text_position[1] + 30)

                    if false_normal:
                        self.draw_text(image, f"False Normal: {', '.join(map(str, false_normal))}", text_position, (255, 0, 0))

                    image_modified = True

            # 이미지 저장
            if image_modified:
                saved_image_count += 1
                output_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_path, image)

        predicted_critical_count = sum(len(result_item['Critical']) for result_item in results)
        predicted_normal_count = sum(len(result_item['Normal']) for result_item in results)

        total_error_count = false_critical_count + false_normal_count + no_gt_critical_count + no_gt_normal_count

        false_critical_percent = (false_critical_count / predicted_critical_count) * 100 if predicted_critical_count > 0 else 0
        false_normal_percent = (false_normal_count / predicted_normal_count) * 100 if predicted_normal_count > 0 else 0
        no_gt_critical_percent = (no_gt_critical_count / predicted_critical_count) * 100 if predicted_critical_count > 0 else 0
        no_gt_normal_percent = (no_gt_normal_count / predicted_normal_count) * 100 if predicted_normal_count > 0 else 0
        model_performance = ((total_images - total_error_count) / max(total_images, 1)) * 100   # TP / TP + FP  => ## Precision

        # wandb에 최종 결과 로깅
        wandb.log({
            "False Critical %": false_critical_percent,
            "False Normal %": false_normal_percent,
            "No GT Critical %": no_gt_critical_percent,
            "No GT Normal %": no_gt_normal_percent,
            "전체 Error 수": total_error_count,
            "Model Performance(precision)": model_performance
        })

        print()
        print(f"전체 예측 결과 수: {total_images}")
        print(f"예측된 Critical 수: {predicted_critical_count}")
        print(f"예측된 Normal 수: {predicted_normal_count}")
        print(f"False Critical 수: {false_critical_count} ({false_critical_percent:.2f}%)")
        print(f"False Normal 수: {false_normal_count} ({false_normal_percent:.2f}%)")
        print(f"No GT Critical 수: {no_gt_critical_count} ({no_gt_critical_percent:.2f}%)")
        print(f"No GT Normal 수: {no_gt_normal_count} ({no_gt_normal_percent:.2f}%)")
        print(f"저장된 Error 이미지 수: {saved_image_count}")
        print(f"전체 Error 수: {total_error_count}")
        print(f"Model Performance: {model_performance:.2f}%")

        print('작업 완료')