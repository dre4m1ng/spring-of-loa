import os
import re
import pandas as pd

def extract_number(f):
    s = re.findall(r"(\d+)", f)
    return (int(s[-1]) if s else -1, f)


def convert_images_to_csv(folder_path, csv_file_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')], key=extract_number)
    df = pd.DataFrame(image_files, columns=['Image Name'])
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    print('CSV 파일이 성공적으로 생성되었습니다.')


def convert_csv_to_json(csv_file_path, json_file_path):
    df = pd.read_csv(csv_file_path)
    df.dropna(how='all', subset=['Critical', 'Normal'], inplace=True)
    df['Critical'] = df['Critical'].apply(lambda x: [int(float(i)) for i in str(x).split('\n') if i] if pd.notnull(x) else [])
    df['Normal'] = df['Normal'].apply(lambda x: [int(float(i)) for i in str(x).split('\n') if i] if pd.notnull(x) else [])
    df['sort_key'] = df['Image Name'].apply(lambda x: extract_number(x)[0])
    df.sort_values(by='sort_key', inplace=True)
    df.drop(columns=['sort_key'], inplace=True)
    df.to_json(json_file_path, orient='records', force_ascii=False)
    
    print('JSON 파일 변환이 완료되었습니다.')