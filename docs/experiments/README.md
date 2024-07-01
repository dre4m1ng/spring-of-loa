# binary
- 모든 30개의 경우의 수를 이진화

# output
- easy: 30 + 30개 이진화 결과
- paddle: 30 + 30개 이진화 결과

# 실험결과
- 실험해 본 모든 사진
- QA: 화질 변화
- GQA: 내 전처리 상한값 상승후 전처리
- posstgaussian: 화질 전처리 후 가우시안 블러

# 30개 경우의수 사진
- 1번 전처리 방식
  ```python
  # 색상에 따른 범위 설정
  if color_name == 'yellow':
      color_lower = np.array([20, 160, 215])
      color_upper = np.array([30, 255, 255]) # [40, 255, 255]
  elif color_name == 'white':
      color_lower = np.array([0, 0, 200])
      color_upper = np.array([180, 55, 255])
  elif color_name == 'yellow+white':
      # 노란색 범위
      yellow_lower = np.array([20, 160, 215])
      yellow_upper = np.array([30, 255, 255]) # [40, 255, 255]
      yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
      # 하얀색 범위
      white_lower = np.array([0, 0, 200])
      white_upper = np.array([180, 55, 255])
      white_mask = cv2.inRange(hsv, white_lower, white_upper)
      # 노란색과 하얀색 마스크 합치기
      mask = cv2.bitwise_or(yellow_mask, white_mask)
  else:
      raise ValueError("지원되지 않는 색상입니다. 'yellow', 'white', 'yellow+white' 중 하나를 선택해주세요.")
  
  # 노란색과 하얀색 이외의 경우에만 마스크 생성
  if color_name != 'yellow+white':
      mask = cv2.inRange(hsv, color_lower, color_upper)
  
  # 모폴로지 연산을 사용한 침식
  kernel = np.ones((3,3), np.uint8)
  eroded_mask = cv2.erode(mask, kernel, iterations=1)
  ```

- 2번 전처리 방식
  ```python
  # 노란색의 HSV 범위를 정의합니다.
  lower_yellow = np.array([20, 160, 215])  # 노란색의 하한값
  upper_yellow = np.array([40, 255, 255])  # 노란색의 상한값

  # HSV 이미지에서 노란색 범위에 해당하는 픽셀을 찾습니다.
  yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

  # 노란색을 제외한 나머지 부분을 검은색으로 대체합니다.
  image[yellow_mask == 0] = [0, 0, 0]
  ```

**1~10**
- "preprocessed_3(1).webp": 1번 전처리
- "preprocessed_3(2).jpg": 2번 전처리
- "preprocessed_3(3).jpg": 화질 저하(450) + 1번 전처리
- "preprocessed_3(4).jpg": 화질 저하(450) + 2번 전처리
- "preprocessed_3(5).jpg": 화질 저하(450) + 가우시안 블러 + 1번 전처리
- "preprocessed_3(6).jpg": 화질 저하(450) + 가우시안 블러 + 2번 전처리
- "preprocessed_3(7).jpg": 가우시안 블러 + 1번 전처리
- "preprocessed_3(8).jpg": 가우시안 블러 + 2번 전처리
- "preprocessed_3(9).jpg": 화질 저하(700) + 1번 전처리
- "preprocessed_3(10).jpg": 화질 저하(700) + 2번 전처리

**가우시안 없이 화질용**
- "preprocessed_3_700temp(1).jpg"
– "preprocessed_3_700(2).jpg"
- "preprocessed_3_750temp(1).jpg"
- "preprocessed_3_750(2).jpg"
- "preprocessed_3_800temp(1).jpg"
- "preprocessed_3_800(2).jpg"
- "preprocessed_3_850temp(1).jpg"
- "preprocessed_3_850(2).jpg"
- "preprocessed_3_900temp(1).jpg"
- "preprocessed_3_900(2).jpg"
- "preprocessed_3_1000temp(1).jpg"
- "preprocessed_3_1000(2).jpg"
- "preprocessed_3_1250temp(1).jpg"
- "preprocessed_3_1250(2).jpg"
- "preprocessed_3_1400temp(1).jpg"
- "preprocessed_3_1400(2).jpg"

**2번 전처리 + 가우시안**
- "preprocessed_3_post_gaussian700.jpg"
- "preprocessed_3_post_gaussian750.jpg"
- "preprocessed_3_post_gaussian800.jpg"
- "preprocessed_3_post_gaussian850.jpg"

# 결론
- 화질 저하는 700 x 700이 가장 좋았음
- 2번 전처리 코드가 값을 더 잘 찾음
- 화질 저하 + 색 추출 + 가우시안 순으로 전처리 하는 방식이 결과값이 좋았음
- 이진화 처리를 한 경우 좀 더 나은 결과를 얻을 수 있었음, 하지만 이진화를 가우시안 처리 전에 하는것이 결과가 더 좋았음
