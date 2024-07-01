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
        �ʱ�ȭ �޼���, ground truth �������� ��θ� �����ϰ� �����͸� �ε��մϴ�.

        Args:
            gt_path (str): Ground Truth ������ ���� ���
        """
        self.gt_path = gt_path
        self.gt_data = self._load_gt_data()
        self.project_name = "ocr"
        self.entity = "sol_of_loa"

        # wanDB init
        wandb.init(project = self.project_name, entity = self.entity, name = run_name)

    def _load_gt_data(self):
        """
        Ground Truth �����͸� �ε��ϴ� ���� �޼���

        Returns:
            list: Ground Truth ������ ����Ʈ
        """
        try:
            with open(self.gt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, OSError) as e:
            print(f"Error reading ground truth data from {self.gt_path}: {e}")
            return None
    
    def _load_results(self, results):
        """
        ��� �����͸� �ε��ϴ� ���� �޼���

        Args:
            results (str or list): ��� ������ ���� ��� �Ǵ� ��� ������ ����Ʈ

        Returns:
            list: ��� ������ ����Ʈ
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
        ��ó�� ��ǥ�� �����ϰ� wandb�� �α��ϴ� �޼���

        Args:
            results (str or list): ��� ������ ���� ��� �Ǵ� ��� ������ ����Ʈ
            critical_operations (list): ũ��Ƽ�� ��ó�� ���� ����Ʈ
            normal_operations (list): �븻 ��ó�� ���� ����Ʈ

        Returns:
            dict: ��ó�� ��ǥ ��ųʸ�
        """
        results = self._load_results(results)
        if results is None:
            return None

        # Ground Truth ���� ���
        total_gt_critical_numbers = 0
        total_gt_normal_numbers = 0

        for gt_item in self.gt_data:
            total_gt_critical_numbers += len(gt_item['Critical'])
            total_gt_normal_numbers += len(gt_item['Normal'])

        total_gt_numbers = total_gt_critical_numbers + total_gt_normal_numbers

        # OCR ����� ������ ���
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

        # ��ǥ ���
        overall_found_gt_ratio = found_gt_count / total_gt_numbers if total_gt_numbers > 0 else 0
        critical_found_gt_ratio = found_critical_count / total_gt_critical_numbers if total_gt_critical_numbers > 0 else 0
        normal_found_gt_ratio = found_normal_count / total_gt_normal_numbers if total_gt_normal_numbers > 0 else 0

        overall_found_ocr_ratio = found_gt_count / total_ocr_numbers if total_ocr_numbers > 0 else 0
        critical_found_ocr_ratio = found_critical_count / total_ocr_critical_numbers if total_ocr_critical_numbers > 0 else 0
        normal_found_ocr_ratio = found_normal_count / total_ocr_normal_numbers if total_ocr_normal_numbers > 0 else 0


        # ��ó�� ���� �α׸� ���̺� ���·� ���
        preprocessing_table = wandb.Table(columns=["Preprocessing Step Type", "Operations"])
        preprocessing_table.add_data("Critical Preprocessing Steps", ", ".join(critical_operations))
        preprocessing_table.add_data("Normal Preprocessing Steps", ", ".join(normal_operations))

        metrics = {
            "Total (OCR�� GT / ��ü GT) damage percent(��ó��)": overall_found_gt_ratio,    # recall
            "Critical (OCR�� GT / ��ü GT) (��ó��)": critical_found_gt_ratio,
            "Normal (OCR�� GT / ��ü GT) (��ó��)": normal_found_gt_ratio,
            "Total (OCR�� GT / ��ü OCR) (��ó��)": overall_found_ocr_ratio,    # Precison
            "Critical (OCR�� GT / ��ü OCR) (��ó��)": critical_found_ocr_ratio,
            "Normal (OCR�� GT / ��ü OCR) (��ó��)": normal_found_ocr_ratio,
            "Preprocessing Steps (��ó�� ����)": preprocessing_table,
        }

        wandb.log(metrics)
        return metrics
    
    def generate_postprocessing_metrics(self, pre_results, post_results):
        """
        ��ó�� ��ǥ�� �����ϰ� wandb�� �α��ϴ� �޼���

        Args:
            pre_results (str or list): ��ó�� �� ��� ������ ���� ��� �Ǵ� ��� ������ ����Ʈ
            post_results (str or list): ��ó�� �� ��� ������ ���� ��� �Ǵ� ��� ������ ����Ʈ

        Returns:
            dict: ��ó�� ��ǥ ��ųʸ�
        """
        pre_results = self._load_results(pre_results)
        post_results = self._load_results(post_results)
        if pre_results is None or post_results is None:
            return None

        # �ʱ�ȭ
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
            "Total ���︸ŭ ������ (��ó��)": total_fp_post / total_fp_pre if total_fp_pre > 0 else 0,  # ���ۼ�Ʈ ������� ���°� ���� �� �� ���� ��ġ
            "Total ����� �� �Ǵ°ǵ� (��ó��)": total_tp_post / total_tp_pre if total_tp_pre > 0 else 0,       # �� ���� �ۼ�Ʈ ���� �� �� ���� ��ġ
        }
        wandb.log(metrics)
        return metrics

    def generate_final_metrics(self, results): 
        """
        ���� �� ��ǥ�� �����ϰ� wandb�� �α��ϴ� �޼���

        Args:
            results (str or list): ��� ������ ���� ��� �Ǵ� ��� ������ ����Ʈ

        Returns:
            dict: ���� �� ��ǥ ��ųʸ�
        """
        results = self._load_results(results)
        if results is None:
            return None

        # Ground Truth ���� ���
        total_gt_critical_damage = 0
        total_gt_normal_damage = 0

        for gt_item in self.gt_data:
            total_gt_critical_damage += sum(gt_item['Critical'])
            total_gt_normal_damage += sum(gt_item['Normal'])

        # OCR ��� ���� ���
        total_ocr_critical_damage = 0
        total_ocr_normal_damage = 0

        for result_item in results:
            total_ocr_critical_damage += sum(result_item['Critical'])
            total_ocr_normal_damage += sum(result_item['Normal'])

        # ��ǥ ���
        overall_damage_match_ratio = (total_ocr_critical_damage + total_ocr_normal_damage) / (total_gt_critical_damage + total_gt_normal_damage) if (total_gt_critical_damage + total_gt_normal_damage) > 0 else 0

        metrics = {
            "Total ������ ���� (OCR / GT)": overall_damage_match_ratio,
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
            else:  # fp_normal�� �ִ� ���
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


# ��� ����
# evaluator = EvaluationMetrics(gt_path='path_to_ground_truth.json')
# preprocessing_metrics = evaluator.generate_preprocessing_metrics(results)
# postprocessing_metrics = evaluator.generate_postprocessing_metrics(results)
# final_metrics = evaluator.generate_final_metrics(results='path_to_results.json')
