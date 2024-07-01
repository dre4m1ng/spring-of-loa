import re
import json
import os
import glob
import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import wandb


class EvaluationMetrics:
    def __init__(self, gt_path, run_name=None):
        """
        초기화 메서드, ground truth 데이터의 경로를 설정하고 데이터를 로드합니다.

        Args:
            gt_path (str): Ground Truth 데이터 파일 경로
        """
        self.gt_path = gt_path
        self.gt_data = self._load_gt_data()
        self.project_name = "ocr"
        self.entity = "sol_of_loa"

        # wanDB init
        wandb.init(project = self.project_name, entity = self.entity, name = run_name)

    def _load_gt_data(self):
        """
        Ground Truth 데이터를 로드하는 내부 메서드

        Returns:
            list: Ground Truth 데이터 리스트
        """
        try:
            with open(self.gt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, OSError) as e:
            print(f"Error reading ground truth data from {self.gt_path}: {e}")
            return None
    
    def _load_results(self, results):
        """
        결과 데이터를 로드하는 내부 메서드

        Args:
            results (str or list): 결과 데이터 파일 경로 또는 결과 데이터 리스트

        Returns:
            list: 결과 데이터 리스트
        """
        if isinstance(results, str):
            try:
                with open(results, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f"Error reading results from {results}: {e}")
                return None
        elif isinstance(results, list):
            return results
        else:
            print("Invalid results format. Expected JSON file path or list.")
            return None
    
    def generate_preprocessing_metrics(self, results, critical_operations, normal_operations):
        """
        전처리 지표를 생성하고 wandb에 로깅하는 메서드

        Args:
            results (str or list): 결과 데이터 파일 경로 또는 결과 데이터 리스트
            critical_operations (list): 크리티컬 전처리 과정 리스트
            normal_operations (list): 노말 전처리 과정 리스트

        Returns:
            dict: 전처리 지표 딕셔너리
        """
        results = self._load_results(results)
        if results is None:
            return None

        # Ground Truth 총합 계산
        total_gt_critical_numbers = 0
        total_gt_normal_numbers = 0

        for gt_item in self.gt_data:
            total_gt_critical_numbers += len(gt_item['Critical'])
            total_gt_normal_numbers += len(gt_item['Normal'])

        total_gt_numbers = total_gt_critical_numbers + total_gt_normal_numbers

        # OCR 결과와 교집합 계산
        found_critical_count = 0
        found_normal_count = 0

        for result_item in results:
            gt_item = next((item for item in self.gt_data if item['Image Name'] == result_item['Image Name']), None)
            if gt_item:
                found_critical_count += len(set(result_item['Critical']).intersection(gt_item['Critical']))
                found_normal_count += len(set(result_item['Normal']).intersection(gt_item['Normal']))

        found_gt_count = found_critical_count + found_normal_count
        total_ocr_numbers = sum(len(result_item['Critical']) + len(result_item['Normal']) for result_item in results)
        total_ocr_critical_numbers = sum(len(result_item['Critical']) for result_item in results)
        total_ocr_normal_numbers = sum(len(result_item['Normal']) for result_item in results)

        # 지표 계산
        overall_found_gt_ratio = found_gt_count / total_gt_numbers if total_gt_numbers > 0 else 0
        critical_found_gt_ratio = found_critical_count / total_gt_critical_numbers if total_gt_critical_numbers > 0 else 0
        normal_found_gt_ratio = found_normal_count / total_gt_normal_numbers if total_gt_normal_numbers > 0 else 0

        overall_found_ocr_ratio = found_gt_count / total_ocr_numbers if total_ocr_numbers > 0 else 0
        critical_found_ocr_ratio = found_critical_count / total_ocr_critical_numbers if total_ocr_critical_numbers > 0 else 0
        normal_found_ocr_ratio = found_normal_count / total_ocr_normal_numbers if total_ocr_normal_numbers > 0 else 0


        # 전처리 과정 로그를 테이블 형태로 기록
        preprocessing_table = wandb.Table(columns=["Preprocessing Step Type", "Operations"])
        preprocessing_table.add_data("Critical Preprocessing Steps", ", ".join(critical_operations))
        preprocessing_table.add_data("Normal Preprocessing Steps", ", ".join(normal_operations))

        metrics = {
            "Total (OCR중 GT / 전체 GT) damage percent(전처리)": overall_found_gt_ratio,    # recall
            "Critical (OCR중 GT / 전체 GT) (전처리)": critical_found_gt_ratio,
            "Normal (OCR중 GT / 전체 GT) (전처리)": normal_found_gt_ratio,
            "Total (OCR중 GT / 전체 OCR) (전처리)": overall_found_ocr_ratio,    # Precison
            "Critical (OCR중 GT / 전체 OCR) (전처리)": critical_found_ocr_ratio,
            "Normal (OCR중 GT / 전체 OCR) (전처리)": normal_found_ocr_ratio,
            "Preprocessing Steps (전처리 순서)": preprocessing_table,
        }

        wandb.log(metrics)
        return metrics
    
    def generate_postprocessing_metrics(self, pre_results, post_results):
        """
        후처리 지표를 생성하고 wandb에 로깅하는 메서드

        Args:
            pre_results (str or list): 후처리 전 결과 데이터 파일 경로 또는 결과 데이터 리스트
            post_results (str or list): 후처리 후 결과 데이터 파일 경로 또는 결과 데이터 리스트

        Returns:
            dict: 후처리 지표 딕셔너리
        """
        pre_results = self._load_results(pre_results)
        post_results = self._load_results(post_results)
        if pre_results is None or post_results is None:
            return None

        # 초기화
        total_fp_pre = 0
        total_fp_post = 0
        total_tp_pre = 0
        total_tp_post = 0

        for gt_item in self.gt_data:
            gt_critical = set(gt_item['Critical'])
            gt_normal = set(gt_item['Normal'])

            pre_item = next((item for item in pre_results if item['Image Name'] == gt_item['Image Name']), None)
            post_item = next((item for item in post_results if item['Image Name'] == gt_item['Image Name']), None)

            if pre_item:
                pre_critical = set(pre_item['Critical'])
                pre_normal = set(pre_item['Normal'])

                total_fp_pre += len(pre_critical - gt_critical) + len(pre_normal - gt_normal)
                total_tp_pre += len(pre_critical & gt_critical) + len(pre_normal & gt_normal)

                if post_item:
                    post_critical = set(post_item['Critical'])
                    post_normal = set(post_item['Normal'])

                    total_fp_post += len(post_critical - gt_critical) + len(post_normal - gt_normal)
                    total_tp_post += len(post_critical & gt_critical) + len(post_normal & gt_normal)

        metrics = {
            "Total 지울만큼 지웠나 (후처리)": total_fp_post / total_fp_pre if total_fp_pre > 0 else 0,  # 몇퍼센트 지운건지 보는것 높을 수 록 좋은 수치
            "Total 지우면 안 되는건데 (후처리)": total_tp_post / total_tp_pre if total_tp_pre > 0 else 0,       # 악 영향 퍼센트 낮을 수 록 좋은 수치
        }
        wandb.log(metrics)
        return metrics

    def generate_final_metrics(self, results): 
        """
        최종 평가 지표를 생성하고 wandb에 로깅하는 메서드

        Args:
            results (str or list): 결과 데이터 파일 경로 또는 결과 데이터 리스트

        Returns:
            dict: 최종 평가 지표 딕셔너리
        """
        results = self._load_results(results)
        if results is None:
            return None

        # Ground Truth 총합 계산
        total_gt_critical_damage = 0
        total_gt_normal_damage = 0

        for gt_item in self.gt_data:
            total_gt_critical_damage += sum(gt_item['Critical'])
            total_gt_normal_damage += sum(gt_item['Normal'])

        # OCR 결과 총합 계산
        total_ocr_critical_damage = 0
        total_ocr_normal_damage = 0

        for result_item in results:
            total_ocr_critical_damage += sum(result_item['Critical'])
            total_ocr_normal_damage += sum(result_item['Normal'])

        # 지표 계산
        overall_damage_match_ratio = (total_ocr_critical_damage + total_ocr_normal_damage) / (total_gt_critical_damage + total_gt_normal_damage) if (total_gt_critical_damage + total_gt_normal_damage) > 0 else 0

        metrics = {
            "Total 데미지 비율 (OCR / GT)": overall_damage_match_ratio,
        }

        wandb.log(metrics)
        return metrics

    def draw_bbox_with_label(self, image, bbox, label, color):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw a filled rectangle for the label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Draw the label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def visualize_errors(self, post_result, yellow_folder_path, white_folder_path, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        post_results = self._load_results(post_result)

        fp_errors = []
        for result_item in post_results:
            image_name = result_item['Image Name']
            gt_item = next((item for item in self.gt_data if item["Image Name"] == image_name), None)
            
            if gt_item is None:
                print(f"No ground truth data found for image: {image_name}")
                continue

            gt_critical = set(gt_item['Critical'])
            gt_normal = set(gt_item['Normal'])
            pred_critical = set(result_item['Critical'])
            pred_normal = set(result_item['Normal'])

            fp_critical = pred_critical - gt_critical
            fp_normal = pred_normal - gt_normal
            if fp_critical or fp_normal:
                fp_errors.append({
                    'image_name': image_name,
                    'fp_critical': fp_critical,
                    'fp_normal': fp_normal,
                    'ocr_results': result_item['OCR_Results']
                })

        for error in fp_errors:
            image_name = error['image_name']
            fp_critical = error['fp_critical']
            fp_normal = error['fp_normal']
            ocr_results = error['ocr_results']

            if fp_critical:
                image_path = os.path.join(yellow_folder_path, image_name)
            else:  # fp_normal만 있는 경우
                image_path = os.path.join(white_folder_path, image_name)

            image = cv2.imread(image_path)
            if image is not None:
                for ocr_result in ocr_results:
                    text, _, bbox = ocr_result
                    value = int(text.replace(',', ''))
                    
                    if value in fp_critical:
                        color = (0, 0, 255)  # Red for critical
                        label = f"FP Critical: {value}"
                    elif value in fp_normal:
                        color = (255, 0, 0)  # Blue for normal
                        label = f"FP Normal: {value}"
                    else:
                        continue  # Skip if not a false positive

                    self.draw_bbox_with_label(image, bbox[0], label, color)

                output_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_path, image)

        print(f"Processed {len(fp_errors)} images with false positives.")


# 사용 예시
# evaluator = EvaluationMetrics(gt_path='path_to_ground_truth.json')
# preprocessing_metrics = evaluator.generate_preprocessing_metrics(results)
# postprocessing_metrics = evaluator.generate_postprocessing_metrics(results)
# final_metrics = evaluator.generate_final_metrics(results='path_to_results.json')
