#data 폴더를 만들어주자.

import cv2
import numpy as np
import time
import os

def center_crop_to_square(img):
    """
    이미지를 최대 정사각형 크기로 center crop.
    """
    h, w, _ = img.shape  # 원본 이미지 크기
    side_length = min(h, w)  # 정사각형의 한 변 길이
    center_x, center_y = w // 2, h // 2  # 중심 좌표

    # Crop 영역 계산
    x1 = max(center_x - side_length // 2, 0)
    y1 = max(center_y - side_length // 2, 0)
    x2 = x1 + side_length
    y2 = y1 + side_length

    return img[y1:y2, x1:x2]

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cnt = 0
old_tmstmp = time.strftime("%Y%m%d_%H%M%S")
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

try:
    while True:
        batch = []  # 4개의 이미지를 임시 저장할 리스트

        # 4개의 이미지를 캡처
        for i in range(4):
            ret, frame = cap.read()
            if not ret:
                print(f"프레임 {i + 1}을 읽을 수 없습니다.")
                break
            
            # Center Crop을 적용하여 최대 정사각형으로 자름
            cropped_frame = center_crop_to_square(frame)

            # 이미지를 224x224로 리사이즈
            resized_frame = cv2.resize(cropped_frame, (224, 224))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            batch.append(resized_frame)
            
            # 화면에 표시
            cv2.imshow(f'Image {i + 1}', cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)  # 짧은 대기

        # 1초 대기
        while len(os.listdir('./data')) > 10:
            print("file num over 10. waiting...")
            time.sleep(1)
            continue

        # 4개의 이미지를 NumPy 배열로 변환 후 통합
        if len(batch) == 4:
            stacked_batch = np.stack(batch).astype(np.float32)/255.  # (4, 224, 224, 3)
            stacked_batch = np.transpose(stacked_batch.reshape(4, 14, 16, 14, 16, 3), (0, 1, 3, 5, 2, 4)).reshape(4, 196, 768)
            # .bin 파일로 저장
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if old_tmstmp == timestamp:
                cnt += 1
            else:
                cnt = 0
            old_tmstmp = timestamp
            filename = f"data/batch_{timestamp}_{cnt}.bin"
            with open(filename, "wb") as f:
                stacked_batch.tofile(f)
            
            print(f"4개의 이미지가 '{filename}'로 저장되었습니다.")

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("캡처 중단")

finally:
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
