import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

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
    
    classes = ['디스트로이어', '워로드', '버서커', '홀리나이트', '슬레이어', '배틀마스터', '인파이터', '기공사',
            '창술사', '스트라이커', '브레이커', '데빌헌터', '블래스터', '호크아이', '스카우터', '건슬링어', '바드',
            '서머너', '아르카나', '소서리스', '데모닉', '블레이드', '리퍼', '소울이터', '도화가', '기상술사']
    
    raids = ['발탄', '비아키스', '쿠크세이튼', '아브렐슈드', '일리아칸', '카멘', '카양겔', '혼돈의 상아탑', '에키드나', '베히모스']

    st.selectbox('직업', classes)
    st.selectbox('레이드', raids)
    st.selectbox('관문', [1, 2, 3, 4])
    
    save_path = './data'
    video_data = st.file_uploader("", type=['jpg', 'csv', 'mp4'])

    if video_data is not None:
        save_video_file(save_path, video_data)
        
        df = pd.read_csv(save_path + '/' + video_data.name)

    if st.button("분석 시작!"):
        # 분석코드
        time = [round(i * 0.08339, 2) for i in range(len(df))]
        critical = sum_dmg(df, 'critical')
        normal = sum_dmg(df, 'normal')
        
        data = {'critical' : critical, 'normal' : normal}
        f_df = pd.DataFrame(data=data, index=time)

        critical_filter = f_df[f_df['critical'] != 0][['critical']]
        normal_filter = f_df[f_df['normal'] != 0][['normal']]
        
        fig = go.Figure()
        # critical graph
        fig.add_trace(go.Scatter(x=critical_filter.index, y=critical_filter['critical'], mode='lines', name='Critical', marker=dict(color='#F1CA36', size=8)))
        
        # normal graph
        fig.add_trace(go.Scatter(x=normal_filter.index, y=normal_filter['normal'], mode='lines', name='Normal', marker=dict(color='#595959', size=8)))
        
        fig.update_layout(
            xaxis=dict(title='time (s)'),
            yaxis=dict(title='dmg')
        )
        
        st.plotly_chart(fig)
        
        # critical 확률 계산
        critical_cnt = dmg_count(f_df, 'critical')
        normal_cnt = dmg_count(f_df, 'normal')
        
        crt_per = round(critical_cnt / (critical_cnt + normal_cnt), 3) * 100
        max_dmgs = int(max(critical_filter['critical']))
        sum_dmgs = int(sum(normal_filter['normal']) + sum(critical_filter['critical']))
        dps = round(sum_dmgs / time[-1], 2)
        
        st.write('크리티컬 확률:', round(critical_cnt / (critical_cnt + normal_cnt), 3) * 100, '%')
        st.write('최고 데미지:', format(max_dmgs, ','))
        st.write('데미지 총합:', format(sum_dmgs, ','))
        st.write('DPS:', format(dps, ','), 'dmg / s')
        
        st.info(f"""
                크리티컬 확률: {crt_per} % \n
                최고 데미지: {format(max_dmgs, ',')} \n
                데미지 총합: {format(sum_dmgs, ',')} \n
                DPS: {format(dps, ',')} dmg / s
                """)
        
if __name__ == "__main__":
    main()