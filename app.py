"""
YOLO 实时目标检测应用
使用 WebRTC 实现摄像头实时检测，支持在 Streamlit Cloud 部署
参考: https://github.com/whitphx/streamlit-webrtc
"""
import av
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from collections import Counter
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import threading
import time
from datetime import datetime
from turn import get_ice_servers

# 页面配置
st.set_page_config(
    page_title="YOLO 实时目标检测",
    page_icon="🎯",
    layout="wide"
)

# 初始化 session_state（用于持久化保存检测结果）
if "detection_history" not in st.session_state:
    st.session_state.detection_history = {
        "current_objects": [],       # 当前帧检测结果
        "all_detections": [],        # 所有检测结果累计
        "frame_count": 0,            # 处理帧数
        "start_time": None,          # 开始时间
        "end_time": None,            # 结束时间
        "class_counts": Counter(),   # 类别累计计数
    }


@st.cache_resource
def load_yolo_model(model_path: str = "yolov8n.pt"):
    """
    加载 YOLO 模型（使用 cache_resource 避免重复加载）
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model


# 全局锁和数据容器（用于线程间共享数据）
# 参考: https://github.com/whitphx/streamlit-webrtc#pull-values-from-the-callback
lock = threading.Lock()
result_container = {"objects": [], "frame_count": 0}


def create_video_callback(model, confidence_threshold):
    """
    创建视频帧回调函数
    使用闭包传递模型和置信度参数
    """
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        # 将 VideoFrame 转换为 numpy 数组
        image = frame.to_ndarray(format="bgr24")
        
        if model is None:
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        
        # 使用 YOLO 进行检测
        results = model(image, conf=confidence_threshold, verbose=False)
        
        # 获取检测到的对象
        detected_objects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])
                detected_objects.append({
                    "class": class_name,
                    "confidence": confidence
                })
        
        # 更新共享容器（线程安全）
        with lock:
            result_container["objects"] = detected_objects
            result_container["frame_count"] += 1
        
        # 在图像上绘制检测结果
        annotated_frame = results[0].plot()
        
        # 使用 PIL 处理以避免内存泄漏（参考官方文档）
        result_image = Image.fromarray(annotated_frame)
        output_array = np.asarray(result_image)
        
        return av.VideoFrame.from_ndarray(output_array, format="bgr24")
    
    return video_frame_callback


def render_realtime_statistics(objects, frame_count):
    """
    渲染实时检测统计（当前帧）
    """
    if not objects:
        st.info("📊 等待检测结果... 请确保摄像头已开启并有物体被检测到")
        return
    
    st.caption(f"🔴 实时检测中 | 已处理 {frame_count} 帧")
    
    # 统计当前帧各类别数量
    class_names = [obj["class"] for obj in objects]
    class_counts = Counter(class_names)
    
    # 显示当前帧统计
    st.success(f"✅ 当前帧检测到 **{len(objects)}** 个对象")
    
    # 当前帧详情
    if objects:
        df_current = pd.DataFrame([
            {"类别": obj["class"], "置信度": f"{obj['confidence']:.2%}"}
            for obj in objects
        ])
        st.dataframe(df_current, use_container_width=True, height=150)


def render_summary_statistics(history):
    """
    渲染检测结果汇总统计
    """
    all_detections = history["all_detections"]
    frame_count = history["frame_count"]
    class_counts = history["class_counts"]
    start_time = history["start_time"]
    end_time = history["end_time"]
    
    if not all_detections:
        st.info("📊 暂无检测结果")
        return
    
    # 计算检测时长
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()
        duration_str = f"{duration:.1f} 秒"
    else:
        duration_str = "未知"
    
    st.caption(f"⏹️ 检测已停止（结果已汇总）")
    
    # 汇总统计卡片
    st.markdown("### 📈 检测汇总")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📷 处理帧数", f"{frame_count}")
    with col2:
        st.metric("🎯 总检测次数", f"{len(all_detections)}")
    with col3:
        st.metric("📦 类别数", f"{len(class_counts)}")
    with col4:
        st.metric("⏱️ 检测时长", duration_str)
    
    st.markdown("---")
    
    # 类别统计图表
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### 📊 各类别检测次数")
        if class_counts:
            df_counts = pd.DataFrame({
                "类别": list(class_counts.keys()),
                "次数": list(class_counts.values())
            })
            # 按次数排序
            df_counts = df_counts.sort_values("次数", ascending=False)
            st.bar_chart(df_counts.set_index("类别"))
    
    with col_chart2:
        st.markdown("##### 📈 各类别平均置信度")
        # 计算各类别平均置信度
        class_confidences = {}
        for det in all_detections:
            cls = det["class"]
            if cls not in class_confidences:
                class_confidences[cls] = []
            class_confidences[cls].append(det["confidence"])
        
        avg_confidences = {
            cls: sum(confs) / len(confs) 
            for cls, confs in class_confidences.items()
        }
        
        if avg_confidences:
            df_conf = pd.DataFrame({
                "类别": list(avg_confidences.keys()),
                "置信度": [round(v, 3) for v in avg_confidences.values()]
            })
            st.bar_chart(df_conf.set_index("类别"))
    
    # 详细统计表格
    st.markdown("##### 📋 类别详细统计")
    
    # 构建详细统计数据
    stats_data = []
    for cls in class_counts.keys():
        confs = [d["confidence"] for d in all_detections if d["class"] == cls]
        stats_data.append({
            "类别": cls,
            "检测次数": class_counts[cls],
            "占比": f"{class_counts[cls] / len(all_detections) * 100:.1f}%",
            "平均置信度": f"{sum(confs) / len(confs):.2%}",
            "最高置信度": f"{max(confs):.2%}",
            "最低置信度": f"{min(confs):.2%}",
        })
    
    df_stats = pd.DataFrame(stats_data)
    df_stats = df_stats.sort_values("检测次数", ascending=False)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    # 饼图显示类别占比
    st.markdown("##### 🥧 类别占比分布")
    import plotly.express as px
    fig = px.pie(
        values=list(class_counts.values()),
        names=list(class_counts.keys()),
        hole=0.4
    )
    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def main():
    """
    主函数
    """
    st.title("🎯 YOLO 实时目标检测")
    st.markdown("使用 YOLOv8 进行实时目标检测，支持摄像头实时检测和统计分析。")
    
    # 侧边栏配置
    st.sidebar.header("⚙️ 设置")
    
    # 模型选择
    model_option = st.sidebar.selectbox(
        "选择模型",
        ["yolov8n.pt"],
        help="选择 YOLO 模型（n=nano，速度最快）"
    )
    
    # 置信度阈值
    confidence_threshold = st.sidebar.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="只显示置信度高于此阈值的检测结果"
    )
    
    # 清除历史按钮
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 清除检测历史", use_container_width=True):
        st.session_state.detection_history = {
            "current_objects": [],
            "all_detections": [],
            "frame_count": 0,
            "start_time": None,
            "end_time": None,
            "class_counts": Counter(),
        }
        # 同时重置全局容器
        with lock:
            result_container["objects"] = []
            result_container["frame_count"] = 0
        st.sidebar.success("✅ 历史已清除")
        st.rerun()
    
    # 显示 ICE 服务器状态
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 网络状态")
    
    ice_servers = get_ice_servers()
    if any("turn:" in str(s.get("urls", [])) for s in ice_servers):
        st.sidebar.success("✅ TURN 服务器已配置")
    else:
        st.sidebar.warning("⚠️ 使用 STUN 服务器（本地测试可用）")
    
    # 加载模型
    with st.spinner("正在加载 YOLO 模型..."):
        model = load_yolo_model(model_option)
    st.sidebar.success(f"✅ 模型已加载: {model_option}")
    
    # 创建回调函数
    video_callback = create_video_callback(model, confidence_threshold)
    
    # 主布局
    col_video, col_stats = st.columns([3, 2])
    
    with col_video:
        st.subheader("📹 实时检测")
        
        # WebRTC 配置 - 使用 video_frame_callback 参数（官方推荐方式）
        ctx = webrtc_streamer(
            key="yolo-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_callback,
            rtc_configuration={"iceServers": ice_servers},
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480}
                },
                "audio": False
            },
            async_processing=True,
        )
    
    with col_stats:
        st.subheader("📊 检测统计")
        
        # 创建占位符用于动态更新
        status_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        # 当视频正在播放时，使用循环持续更新统计
        if ctx.state.playing:
            status_placeholder.success("🟢 摄像头已连接，正在检测...")
            
            # 记录开始时间
            if st.session_state.detection_history["start_time"] is None:
                st.session_state.detection_history["start_time"] = datetime.now()
            
            # 使用循环持续更新统计信息
            while ctx.state.playing:
                with lock:
                    objects = result_container["objects"].copy()
                    frame_count = result_container["frame_count"]
                
                # 累积保存检测结果
                if objects:
                    st.session_state.detection_history["current_objects"] = objects
                    st.session_state.detection_history["frame_count"] = frame_count
                    
                    # 累积所有检测结果
                    st.session_state.detection_history["all_detections"].extend(objects)
                    
                    # 更新类别计数
                    for obj in objects:
                        st.session_state.detection_history["class_counts"][obj["class"]] += 1
                
                with stats_placeholder.container():
                    render_realtime_statistics(objects, frame_count)
                
                # 短暂休眠，避免过于频繁的更新
                time.sleep(0.5)
            
            # 循环结束，记录结束时间
            st.session_state.detection_history["end_time"] = datetime.now()
            
        else:
            status_placeholder.info("👆 点击 'START' 按钮开启摄像头")
            
            # 从 session_state 读取保存的历史结果
            history = st.session_state.detection_history
            
            with stats_placeholder.container():
                if history["all_detections"]:
                    # 显示汇总统计
                    render_summary_statistics(history)
                else:
                    st.info("📊 请先开启摄像头进行检测")
    
    # 使用说明
    with st.expander("📖 使用说明", expanded=False):
        st.markdown("""
        ### 如何使用
        1. **点击 START 按钮** 开启摄像头
        2. **允许浏览器访问摄像头** 权限
        3. **等待连接建立** 可能需要几秒钟
        4. **查看检测结果** 实时显示在视频上
        5. **统计图表会自动更新**
        6. **停止后显示汇总统计** 包含所有检测结果
        
        ### 统计说明
        - **实时模式**: 显示当前帧检测结果
        - **汇总模式**: 停止后显示完整统计，包括：
          - 总检测次数和处理帧数
          - 各类别检测次数和占比
          - 平均/最高/最低置信度
          - 类别占比饼图
        
        ### 注意事项
        - 首次使用需要下载模型文件
        - 需要稳定的网络连接
        - 建议使用 Chrome/Edge 浏览器
        """)
    
    # 部署信息
    with st.expander("🚀 部署信息", expanded=False):
        st.markdown("""
        ### Streamlit Cloud 部署
        
        如需在 Streamlit Cloud 上部署，请设置以下环境变量：
        - `TWILIO_ACCOUNT_SID`: Twilio Account SID
        - `TWILIO_AUTH_TOKEN`: Twilio Auth Token
        
        ### 获取 Twilio 凭证
        1. 访问 [Twilio Console](https://www.twilio.com/)
        2. 注册/登录账号
        3. 获取 Account SID 和 Auth Token
        
        ### 参考项目
        - [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
        - [style-transfer-web-app](https://github.com/whitphx/style-transfer-web-app)
        """)


if __name__ == "__main__":
    main()
