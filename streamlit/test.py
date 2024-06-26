import cv2
import streamlit as st
import os
import tempfile
import math
from multiprocessing import Pool
import plotly.express as px
import plotly.graph_objs as go
from lostark import lostark_api, ExtractDmg
from Model import ExtractDmg
import json
from PIL import Image

def process_frame(frame_data):
    frame_number, frame = frame_data
    model = ExtractDmg.Model(frame)
    cri = model.img_preprocessing(type='critical')
    nor = model.img_preprocessing(type='normal')
    return frame_number, cri, nor

def process_frames_chunk(frames_chunk, process_count):
    with Pool(process_count) as pool:
        results = pool.imap_unordered(process_frame, frames_chunk)
        return list(results)

# streamlit main
def main():
    # local test
    tumb_img = Image.open('./image/lostark_thumb.png')
    # page thumbnail
    # tumb_img = Image.open('./streamlit/image/lostark_thumb.png')
    st.image(tumb_img)
    st.markdown("## Lost ARK 데미지 영상 분석기")

    raids = {'발탄': 'valtan', '비아키스': 'vykas', '쿠크세이튼': 'kakul', '아브렐슈드': 'brel',
             '일리아칸': 'akkan', '카멘': 'thaemaine', '카양겔': 'kayangel', '상아탑': 'ivory',
             '에키드나': 'echidna', '베히모스': 'behemoth'}
    raids_ls = list(raids.keys())
    
    diff = {'노말': 'normal', '하드': 'hard', '헬': 'hell'}
    diff_ls = list(diff.keys())
    
    with st.form(key='my_form'):
        api_key = st.text_input(label='API Key')
        char_name = st.text_input(label='닉네임')
        raid = st.selectbox('레이드', raids_ls)
        diffi = st.selectbox('난이도', diff_ls)
        gate = st.selectbox('관문', [1, 2, 3, 4])

        video_data = st.file_uploader("", type=['mp4'])
        
        if video_data is not None:
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, video_data.name)
            
            with open(video_path, 'wb') as f:
                f.write(video_data.getbuffer())
            
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # progress_bar = st.progress(0)
            
            frames = []
            result = []
            frame_number = 0
            frame_interval = 10

            while frame_number < total_frames:
                ret, frame = video.read()
                if not ret:
                    break

                if frame_number % frame_interval == 0:
                    frame = cv2.resize(frame, dsize=(720, 480))
                    frames.append((frame_number, frame))
                
                if len(frames) == 2000:
                    process = process_frames_chunk(frames, 4)
                    result += process
                    frames = []
                
                frame_number += 1
            
            if frames:
                process = process_frames_chunk(frames, 4)
                result += process
            
            result.sort(key=lambda x: x[0])
            
            cri_cnt = 0
            nor_cnt = 0

            cri_ls = []
            nor_ls = []
            time_ls = []
            for i in range(len(result)):
                time = round(result[i][0] * 0.01666667, 2)
                cri_dmg = result[i][1]
                nor_dmg = result[i][2]
                
                time_ls.append(time)
                nor_ls.append(nor_dmg)
                cri_ls.append(cri_dmg)
                
                if cri_dmg > 0:
                    cri_cnt += 1
                if nor_dmg > 0:
                    nor_cnt += 1
            
        submit_button = st.form_submit_button(label='분석 시작!')

    if submit_button:
        # LostARK API를 이용해서 유저 정보 받아옴
        lostark = lostark_api.API(char_name, 'bearer ' + api_key)

        user_profile = lostark.profile()

        user_class = user_profile.get('CharacterClassName')
        user_lv = user_profile.get('ItemAvgLevel')

        user_equip = lostark.equipment()

        el_sum = 0
        chowal = 0
        if float(user_lv.replace(',', '')) >= 1600:
            for i in range(6):
                eq_json = json.loads(user_equip[i].get('Tooltip'))

                eq_dict = {}
                for key, value in eq_json.items():
                    eq_dict[key] = value

                if i != 0:
                    try:
                        el01_lv = int(eq_dict.get('Element_010').get('value').get('Element_000').get('contentStr').get('Element_000').get('contentStr').split('</FONT>')[1][-1])
                        el02_lv = int(eq_dict.get('Element_010').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</FONT>')[1][-1])
                    except (AttributeError, ValueError):
                        try:
                            el01_lv = int(eq_dict.get('Element_009').get('value').get('Element_000').get('contentStr').get('Element_000').get('contentStr').split('</FONT>')[1][-1])
                            el02_lv = int(eq_dict.get('Element_009').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</FONT>')[1][-1])
                        except (AttributeError, ValueError):
                            try:
                                el01_lv = int(eq_dict.get('Element_008').get('value').get('value').get('Element_000').get('contentStr').get('Element_000').get('contentStr').split('</FONT>')[1][-1])
                                el02_lv = int(eq_dict.get('Element_008').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</FONT>')[1][-1])
                            except AttributeError:
                                pass
                    el_sum += el01_lv + el02_lv

                if float(user_lv.replace(',', '')) >= 1620 and i == 1:
                    try:
                        chowal = int(eq_dict.get('Element_009').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</img>')[-1].split('개')[0])
                    except:
                        try:
                            chowal = int(eq_dict.get('Element_008').get('value').get('contentStr').get('Element_001').get('contentStr').split('</img>')[-1].split('개')[0])
                        except:
                            pass

        st.markdown("**[케릭터 정보]**")
        st.write('레벨:', user_lv)
        st.write('직업:', user_class)
        st.write('엘릭서:', el_sum)
        st.write('초월:', chowal)
        
        fig = go.Figure()
        # critical graph
        fig.add_trace(go.Scatter(x=time_ls, y=cri_ls, mode='lines', name='Critical', marker=dict(color='#F1CA36', size=8)))
        
        # normal graph
        fig.add_trace(go.Scatter(x=time_ls, y=nor_ls, mode='lines', name='Normal', marker=dict(color='#595959', size=8)))
        
        fig.update_layout(
            xaxis=dict(title='time (s)'),
            yaxis=dict(title='dmg')
        )
        
        st.plotly_chart(fig)
        
        sum_dmgs = sum(cri_ls) + sum(nor_ls)
        if cri_cnt + nor_cnt > 0:
            crt_per = round(cri_cnt / (cri_cnt + nor_cnt), 3) * 100
        else:
            crt_per = 0
        if cri_ls:
            max_dmgs = max(cri_ls)
        else:
            max_dmgs = 0
        if time_ls:
            dps = round(sum_dmgs / time_ls[-1], 2)
        else:
            dps = 0
        
        with st.form(key='final'):
            st.write('데미지 총합:', sum_dmgs)
            st.write('크리티컬 확률 (%):', crt_per)
            st.write('최고 데미지:', max_dmgs)
            st.write('데미지 총합:', sum_dmgs)
            st.write('DPS (dmg / s):', dps)
            
            st.form_submit_button('분석 완료!')
        
if __name__ == "__main__":
    main()
