import streamlit as st
import cv2
import av
import numpy as np
import threading
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ----------------------------------
# 页面配置
# ----------------------------------
st.set_page_config(
    page_title="YOLO 实时缺陷检测",
    layout="wide"
)

st.title("🔍 YOLO 摄像头实时缺陷检测（Streamlit Cloud）")

# ----------------------------------
# 线程安全共享数据
# ----------------------------------
lock = threading.Lock()
shared_defect_info = {
    "count": 0,
    "labels": []
}

# ----------------------------------
# 加载 YOLO 缺陷模型
# ----------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")   # ← 换成你的缺陷模型

model = load_yolo()
class_names = model.names

# ----------------------------------
# YOLO 视频处理器
# ----------------------------------
class DefectDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 降分辨率（提升 Cloud 稳定性）
        img = cv2.resize(img, (640, 480))

        # YOLO 推理
        results = model(img, conf=0.4, verbose=False)[0]

        detected_labels = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                detected_labels.append(label)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 红色框表示缺陷
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # 写入共享数据（不要用 session_state）
        with lock:
            shared_defect_info["count"] = len(detected_labels)
            shared_defect_info["labels"] = detected_labels

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------------
# WebRTC 摄像头
# ----------------------------------
webrtc_ctx = webrtc_streamer(
    key="defect-detect",
    video_processor_factory=DefectDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ----------------------------------
# UI 面板
# ----------------------------------
col1, col2 = st.columns(2)

count_placeholder = col1.empty()
label_placeholder = col2.empty()

if webrtc_ctx.state.playing:
    with lock:
        count = shared_defect_info["count"]
        labels = shared_defect_info["labels"]

    count_placeholder.metric("缺陷数量", count)

    label_placeholder.subheader("检测到的缺陷类型")
    if labels:
        label_placeholder.write(list(set(labels)))
    else:
        label_placeholder.write("未检测到缺陷")
else:
    st.info("▶️ 点击 Start 启动摄像头")
