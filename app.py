import streamlit as st
import cv2
import av
import threading
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="YOLO 实时目标检测",
    page_icon="🚀",
    layout="wide"
)

# -------------------------
# 全局共享变量（线程安全）
# -------------------------
lock = threading.Lock()
shared_data = {
    "num_objects": 0,
    "labels": [],
    "frame_count": 0,  # 用于调试：跟踪处理的帧数
    "last_error": None,  # 用于调试：记录最后的错误
    "processor_initialized": False,  # 用于调试：确认 VideoProcessor 是否被创建
    "processor_error": None  # 用于调试：记录 VideoProcessor 初始化错误
}

# -------------------------
# 加载 YOLO（必须在全局）
# -------------------------
@st.cache_resource
def load_model():
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()

# 如果模型加载失败，显示错误信息
if model is None:
    st.error("⚠️ 无法加载YOLO模型，请检查模型文件是否存在。")
    st.stop()

# -------------------------
# Video Processor
# -------------------------
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        # 在 VideoProcessor 中加载模型（使用缓存避免重复加载）
        try:
            self.model = YOLO("yolov8n.pt")
            print("YOLOProcessor 初始化成功，模型已加载")
        except Exception as e:
            print(f"YOLOProcessor 初始化失败: {e}")
            self.model = None
        
        # 调试：确认 VideoProcessor 被创建
        with lock:
            shared_data["processor_initialized"] = True
            shared_data["processor_error"] = None if self.model else "模型加载失败"
    
    def recv(self, frame):
        try:
            # 检查模型是否已加载
            if self.model is None:
                print("警告：模型未加载，跳过检测")
                return frame
            
            # 更新帧计数器（用于调试）
            with lock:
                shared_data["frame_count"] += 1
                shared_data["last_error"] = None
                # 每100帧打印一次，避免过多输出
                if shared_data["frame_count"] % 100 == 0:
                    print(f"已处理 {shared_data['frame_count']} 帧")
            
            # 转换帧为 numpy 数组
            img = frame.to_ndarray(format="bgr24")
            
            # 检查图像是否有效
            if img is None or img.size == 0:
                return frame
            
            # YOLO 推理（关闭 verbose，使用更快的推理设置）
            results = self.model(img, verbose=False, conf=0.25)[0]
            
            labels = []
            
            # 处理检测结果
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    try:
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        labels.append(label)
                        
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 绘制检测框
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 绘制标签文本
                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                    except Exception:
                        # 单个框处理失败，继续处理下一个
                        continue
            
            # 写入共享数据（线程安全）
            with lock:
                shared_data["num_objects"] = len(labels)
                shared_data["labels"] = labels.copy()  # 复制列表避免引用问题
            
            # 返回处理后的帧
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # 错误处理：如果处理失败，返回原始帧并记录错误
            import traceback
            error_msg = f"VideoProcessor recv 错误: {e}"
            print(error_msg)
            print(traceback.format_exc())
            
            # 记录错误到共享数据
            with lock:
                shared_data["last_error"] = str(e)
            
            return frame

# -------------------------
# UI
# -------------------------
st.title("🚀 YOLO 实时目标检测（Streamlit Cloud 可用）")
st.markdown("---")

# 添加说明信息
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    1. 点击下方的 **▶️ START** 按钮启动摄像头
    2. 允许浏览器访问摄像头权限
    3. 系统将实时检测画面中的目标对象
    4. 检测结果会显示在右侧统计面板中
    """)

# 调试：显示 webrtc_streamer 配置
st.write("🔧 调试信息：")
st.write(f"- 模型已加载: {model is not None}")
st.write(f"- VideoProcessor 类: {YOLOProcessor}")

# 配置 RTC（用于 WebRTC 连接）
rtc_configuration = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# 兼容当前安装的 streamlit-webrtc 版本（0.44.0 使用的是 video_transformer_factory）
# 这是统计信息始终为 0 的根本原因：之前传的是 video_processor_factory，库根本没有创建 YOLOProcessor
webrtc_ctx = webrtc_streamer(
    key="yolo",
    video_transformer_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=rtc_configuration,
)

# 显示 webrtc 状态
st.write(f"- WebRTC 状态: {webrtc_ctx.state}")
st.write(f"- 是否正在播放: {webrtc_ctx.state.playing}")

st.markdown("---")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 检测统计")
    metric_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("🏷️ 检测到的标签")
    label_placeholder = st.empty()

# -------------------------
# 主线程 UI 更新（不依赖 playing 状态，直接读取检测数据）
# -------------------------
# 从共享数据中读取最新的检测结果（线程安全）
with lock:
    num = shared_data["num_objects"]
    labels = shared_data["labels"].copy() if shared_data["labels"] else []
    frame_count = shared_data.get("frame_count", 0)
    last_error = shared_data.get("last_error", None)
    processor_initialized = shared_data.get("processor_initialized", False)

# 判断是否有视频流在处理（通过 frame_count 判断）
has_video_stream = frame_count > 0

# ========== 始终显示统计信息 ==========
# 更新UI显示 - 对象数量
metric_placeholder.metric("检测到的对象数量", num)

# 调试信息：显示原始数据（帮助排查问题）
with st.expander("🔍 调试信息（点击展开）", expanded=False):
    import time
    current_time = time.strftime("%H:%M:%S")
    st.write(f"**更新时间**: {current_time}")
    st.write(f"- 检测到的对象数量: {num}")
    st.write(f"- 标签列表: {labels}")
    st.write(f"- 已处理帧数: {frame_count}")
    st.write(f"- 是否有视频流: {has_video_stream}")
    st.write(f"- Processor 已初始化: {processor_initialized}")
    st.write(f"- 刷新计数器: {st.session_state.get('refresh_counter', 0)}")
    if last_error:
        st.write(f"- 最后错误: {last_error}")
    
    # 显示 shared_data 的原始内容（用于调试）
    st.write("**shared_data 原始内容:**")
    st.json({
        "num_objects": shared_data.get("num_objects", 0),
        "labels": shared_data.get("labels", []),
        "frame_count": shared_data.get("frame_count", 0),
        "last_error": shared_data.get("last_error", None),
        "processor_initialized": shared_data.get("processor_initialized", False)
    })

# 显示视频流状态
if has_video_stream:
    st.success(f"✅ 视频流运行中 - 已处理 {frame_count} 帧")
    if last_error:
        st.error(f"⚠️ 检测到错误: {last_error}")
else:
    if processor_initialized:
        st.info("⏳ VideoProcessor 已初始化，等待视频流启动...")
    else:
        st.warning("⏳ VideoProcessor 尚未初始化...")

# 显示检测结果
if num > 0:
    status_placeholder.success("✅ 检测到目标对象")
    label_text = "**检测到的对象：**\n\n"
    # 显示标签列表（去重并统计数量）
    unique_labels = list(set(labels))
    for i, label in enumerate(unique_labels, 1):
        count = labels.count(label)
        label_text += f"{i}. {label} (x{count})\n"
    label_placeholder.markdown(label_text)
else:
    if has_video_stream:
        status_placeholder.info("🔍 等待检测目标...")
        label_placeholder.write("暂无检测到对象（请确保画面中有可检测的对象，如人、手机、杯子等）")
    else:
        status_placeholder.info("💡 点击上方 ▶️ START 按钮启动摄像头")
        label_placeholder.write("等待启动摄像头...")

# ========== 自动刷新机制 ==========
# 关键修复：无论初始状态如何，都要定期刷新以检查是否有新的数据
# 使用 session_state 跟踪刷新次数和上次的 frame_count
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0
if "last_frame_count" not in st.session_state:
    st.session_state.last_frame_count = frame_count

st.session_state.refresh_counter += 1

# 检查 frame_count 是否有变化（说明 recv 在工作）
frame_count_changed = frame_count != st.session_state.last_frame_count
if frame_count_changed:
    st.session_state.last_frame_count = frame_count

# 根据是否有视频流决定刷新频率
if has_video_stream:
    # 有视频流时，每3秒刷新一次（更频繁）
    refresh_interval = 6  # 约3秒
    status_msg = f"💡 自动刷新中... 当前检测到 {num} 个对象，已处理 {frame_count} 帧"
else:
    # 没有视频流时，每5秒检查一次
    refresh_interval = 10  # 约5秒
    status_msg = "💡 定期检查中... 等待视频流启动"

# 定期刷新页面以更新统计信息
if st.session_state.refresh_counter >= refresh_interval:
    st.session_state.refresh_counter = 0
    st.rerun()

# 添加手动刷新按钮
col_refresh = st.columns([1, 1, 1])
with col_refresh[1]:
    if st.button("🔄 立即刷新统计", key="refresh_stats_btn"):
        st.rerun()

# 显示状态提示
st.caption(status_msg)
