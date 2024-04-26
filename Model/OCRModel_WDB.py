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
        OCRModel Ŭ������ �ʱ�ȭ�մϴ�.

        Args:
            lang (str, optional): OCR ���� ���. �⺻���� 'korean'�Դϴ�.
            use_gpu (bool, optional): GPU ��� ����. �⺻���� True�Դϴ�.
            project_name (str, optional): wandb ������Ʈ �̸�. �⺻���� 'OCR'�Դϴ�.
            run_name (str, optional): wandb ���� �̸�. �⺻���� None�Դϴ�.
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

        # wanDB init
        wandb.init(project=project_name, entity="sol_of_loa", name=run_name)

    def process_images(self, yellow_folder_path, white_folder_path, confidence_threshold=0.8):
        """
        ����� ������ �Ͼ�� ������ �̹������� �ؽ�Ʈ�� �����ϰ� ����� ��ȯ�մϴ�.

        Args:
            yellow_folder_path (str): ����� �̹����� �ִ� ���� ���
            white_folder_path (str): �Ͼ�� �̹����� �ִ� ���� ���
            confidence_threshold (float, optional): �ؽ�Ʈ �ν� �ŷڵ� �Ӱ谪. �⺻���� 0.8�Դϴ�.

        Returns:
            list: �� �̹������� ����� ���� ������ ���Ե� ��ųʸ� ����Ʈ
        """
        results = []
        all_critical_numbers = set()
        all_normal_numbers = set()

        # ����� �̹��� ó��
        self._process_folder(yellow_folder_path, results, all_critical_numbers, all_normal_numbers, is_yellow=True, confidence_threshold=confidence_threshold)

        # �Ͼ�� �̹��� ó��
        self._process_folder(white_folder_path, results, all_critical_numbers, all_normal_numbers, is_yellow=False, confidence_threshold=confidence_threshold)

        # Critical�� Normal�� ��� �� ����Ʈ�� ��� ����
        filtered_results = [result for result in results if result['Critical'] or result['Normal']]

        return filtered_results

    def _process_folder(self, folder_path, results, all_critical_numbers, all_normal_numbers, is_yellow, confidence_threshold):
        """
        ������ �̹������� �ؽ�Ʈ�� �����ϰ� ����� �����մϴ�.

        Args:
            folder_path (str): �̹��� ���� ���
            results (list): ����� ������ ����Ʈ
            all_critical_numbers (set): ��� Critical ���� ������ set
            all_normal_numbers (set): ��� Normal ���� ������ set
            is_yellow (bool): ����� �̹������� ����
            confidence_threshold (float): �ؽ�Ʈ �ν� �ŷڵ� �Ӱ谪
        """
        image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        image_paths.sort(key=self.extract_number)

        for image_path in image_paths:
            try:
                # �̹��� �ε�
                image = cv2.imread(image_path)

                # �̹������� �ؽ�Ʈ ����
                result = self.ocr.ocr(image, cls=True)

                # �̹��� ���� �̸� ����
                image_name = os.path.basename(image_path)

                critical_numbers = []
                normal_numbers = []
                ocr_results = [] # OCR ����� ������ ����Ʈ

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
                                ocr_results.append((text, confidence, box)) # OCR ��� ����

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
        ���� �̸����� ���ڸ� �����մϴ�.

        Args:
            filename (str): ���� �̸�

        Returns:
            int: ����� ������ ����, ���ڰ� ������ -1 ��ȯ
        """
        s = re.findall(r'\d+', filename)
        return int(s[-1]) if s else -1

    def save_results(self, results, output_path):
        """
        ����� JSON ���Ϸ� �����մϴ�.

        Args:
            results (list): �� �̹������� ����� ���� ������ ���Ե� ��ųʸ� ����Ʈ
            output_path (str): ��� ���� ���
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
        ����� Ground Truth �����Ϳ� ���Ͽ� ���е�, ������, ��Ȯ���� ����մϴ�.

        Args:
            results (list): �� �̹������� ����� ���� ������ ���Ե� ��ųʸ� ����Ʈ
            gt_path (str): Ground Truth ������ ���� ���

        Returns:
            tuple: Critical�� Normal�� ���� ���е�, ������, ��Ȯ��
        """
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except (IOError, OSError) as e:
            print(f"Error reading ground truth data from {gt_path}: {e}")
            return None
        
        # results ������ Ÿ�� Ȯ��
        if isinstance(results, str):
            # JSON ���� ����� ��� ���Ͽ��� �о����
            try:
                with open(results, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f"Error reading results from {results}: {e}")
                return None
        
        elif isinstance(results, list):
            # ����Ʈ�� ��� �״�� ���
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

                # Critical ���� ��
                correct_critical_count += sum(1 for x in pred_critical if x in gt_critical)
                false_critical_count += sum(1 for x in pred_critical if x not in gt_critical)
                false_negative_critical_count += sum(1 for x in gt_critical if x not in pred_critical)

                # Normal ���� ��
                correct_normal_count += sum(1 for x in pred_normal if x in gt_normal)
                false_normal_count += sum(1 for x in pred_normal if x not in gt_normal)
                false_negative_normal_count += sum(1 for x in gt_normal if x not in pred_normal)
            else:
                false_critical_count += len(gt_critical)
                false_normal_count += len(gt_normal)
                false_negative_critical_count += len(gt_critical)
                false_negative_normal_count += len(gt_normal)
            
            # wandb�� ���� �� �α�
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

        # ��ü ��Ȯ�� ���
        total_correct_count = correct_critical_count + correct_normal_count
        total_numbers = total_critical_numbers + total_normal_numbers
        total_accuracy = total_correct_count / total_numbers if total_numbers > 0 else 0

        # wandb�� ��Ʈ�� �α�
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
        ������� Critical�� Normal �������� �հ踦 ����մϴ�.

        Args:
            results (list): �� �̹������� ����� ���� ������ ���Ե� ��ųʸ� ����Ʈ

        Returns:
            tuple: Critical ������ �հ�, Normal ������ �հ�, ��ü ������ �հ�
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

        # results ������ Ÿ�� Ȯ��
        if isinstance(results, str):
            # JSON ���� ����� ��� ���Ͽ��� �о����
            try:
                with open(results, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print(f"Error reading results from {results}: {e}")
                return None
        
        elif isinstance(results, list):
            # ����Ʈ�� ��� �״�� ���
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

            # Ground truth �����Ϳ��� �ش� �̹����� ���� ��������
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

            # �̹��� ����
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

        # wandb�� ���� ��� �α�
        wandb.log({
            "False Critical %": false_critical_percent,
            "False Normal %": false_normal_percent,
            "No GT Critical %": no_gt_critical_percent,
            "No GT Normal %": no_gt_normal_percent,
            "��ü Error ��": total_error_count,
            "Model Performance(precision)": model_performance
        })

        print()
        print(f"��ü ���� ��� ��: {total_images}")
        print(f"������ Critical ��: {predicted_critical_count}")
        print(f"������ Normal ��: {predicted_normal_count}")
        print(f"False Critical ��: {false_critical_count} ({false_critical_percent:.2f}%)")
        print(f"False Normal ��: {false_normal_count} ({false_normal_percent:.2f}%)")
        print(f"No GT Critical ��: {no_gt_critical_count} ({no_gt_critical_percent:.2f}%)")
        print(f"No GT Normal ��: {no_gt_normal_count} ({no_gt_normal_percent:.2f}%)")
        print(f"����� Error �̹��� ��: {saved_image_count}")
        print(f"��ü Error ��: {total_error_count}")
        print(f"Model Performance: {model_performance:.2f}%")

        print('�۾� �Ϸ�')