import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="WebRTC Test", layout="wide")

st.title("📷 WebRTC 摄像头连通性测试")

webrtc_ctx = webrtc_streamer(
    key="camera-test",
    mode="SENDRECV",
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

st.write("WebRTC state:", webrtc_ctx.state)
