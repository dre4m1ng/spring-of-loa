import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import json
from lostark import lostark_api
from Model import OCRModel, Preprocesser_utils

# 영상 업로드
def save_video_file(directory, file):    
    save_path = os.path.join(directory, file.name)
    with open(save_path, 'wb') as f:
        f.write(file.getvalue())
    return st.success("저장 완료!")

# 분석 코드

# 데미지 계산
def sum_dmg(df, type):
    """
    df: 데이터 프레임
    type: critical, normal
    """
    dmg = []
    n = len(df)
    for i in range(n):
        val = df.loc[i][type]
        if isinstance(val, str) and '\n' in val:
            sum_dmg = sum(float(num) for num in val.split('\n') if num.isdigit())
            dmg.append(sum_dmg)
        elif isinstance(val, str) and val.isdigit():
            dmg.append(int(val))
        else:
            dmg.append(0)
    return dmg

def dmg_count(df, type):
    cnt = list(df[type])
    filter_cnt = 0
    
    for _ in cnt:
        if _ > 0:
            filter_cnt += 1
    return filter_cnt

# streamlit main
def main():
    # page thumbnail
    st.image('./image/lostark_thumb.png')
    st.markdown("## Lost ARK 데미지 영상 분석기")

    raids = ['발탄', '비아키스', '쿠크세이튼', '아브렐슈드', '일리아칸', '카멘', '카양겔', '상아탑', '에키드나', '베히모스']
    
    with st.form(key='my_form'):
        api_key = st.text_input(label='API Key')
        char_name = st.text_input(label='닉네임')
        raid = st.selectbox('레이드', raids)
        diffi = st.selectbox('난이도', ['노말', '하드', '헬'])
        gate = st.selectbox('관문', [1, 2, 3, 4])

        video_data = st.file_uploader("", type=['jpg', 'csv', 'mp4'])

        if video_data is not None:
            video_name = f'{video_data.name[:-4]}'
            save_path = './data/' + video_name
            video_path = './data/' + video_data.name
            # 실행하고 싶은 작업 리스트
            save_video_file('./data', video_data)
            
            vtf = Preprocesser_utils.VideoPreprocessor(video_path=video_path, save_folder=save_path, save_name_prefix=video_name)
            vtf.preprocess_video()
            
            critical = save_path + '/cri'
            normal = save_path + '/nor'
            model = OCRModel.Model()
            model_result = model.process_images(yellow_folder_path=critical, white_folder_path=normal)
            
            tcd, tnd, td = model.calculate_total_damage(model_result)
            
        submit_button = st.form_submit_button(label='분석 시작!')

    if submit_button:
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
                                el01_lv = int(eq_dict.get('Element_008').get('value').get('Element_000').get('contentStr').get('Element_000').get('contentStr').split('</FONT>')[1][-1])
                                el02_lv = int(eq_dict.get('Element_008').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</FONT>')[1][-1])
                            except AttributeError:
                                pass
                    el_sum += el01_lv + el02_lv

                if float(user_lv.replace(',', '')) >= 1620 and i == 1:
                    try:
                        chowal = int(eq_dict.get('Element_009').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</img>')[-1].split('개')[0])
                    except:
                        try:
                            chowal = int(eq_dict.get('Element_008').get('value').get('Element_000').get('contentStr').get('Element_001').get('contentStr').split('</img>')[-1].split('개')[0])
                        except:
                            pass

        st.markdown("**[케릭터 정보]**")
        st.write('레벨:', user_lv)
        st.write('직업:', user_class)
        st.write('엘릭서:', el_sum)
        st.write('초월:', chowal)
        
        with st.form(key='final'):
            st.write('데미지 총합:', td)
            
            st.form_submit_button('분석 완료!')
        
if __name__ == "__main__":
    main()