# 🎯 YOLO 实时目标检测 Web 应用

基于 YOLOv8 和 Streamlit 的实时目标检测应用，支持通过浏览器摄像头进行实时检测，并展示统计图表。可部署到 Streamlit Cloud。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)

## ✨ 功能特点

- 🎥 **实时摄像头检测** - 使用 WebRTC 技术，直接在浏览器中捕获摄像头视频流
- 🤖 **YOLOv8 目标检测** - 使用最新的 YOLOv8 模型进行高效目标检测
- 📊 **实时统计图表** - 动态展示检测对象数量和置信度分布
- ☁️ **云端部署支持** - 完全兼容 Streamlit Cloud 部署
- 🔧 **可调参数** - 支持实时调整置信度阈值

## 📸 应用截图

```
┌─────────────────────────────────────────────────────────────┐
│  🎯 YOLO 实时目标检测                                        │
├─────────────────────────────────┬───────────────────────────┤
│                                 │  📊 实时统计               │
│     📹 实时检测                  │  ├─ 各类别数量柱状图       │
│     ┌─────────────────────┐    │  ├─ 平均置信度柱状图       │
│     │                     │    │  └─ 检测详情表格           │
│     │   [摄像头画面]       │    │                           │
│     │   带检测框标注       │    │                           │
│     │                     │    │                           │
│     └─────────────────────┘    │                           │
│         [START] [STOP]         │                           │
└─────────────────────────────────┴───────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 支持 WebRTC 的浏览器（Chrome、Edge、Firefox）
- 摄像头设备

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/yolo-streamlit-cloud.git
cd yolo-streamlit-cloud

# 安装依赖
pip install -r requirements.txt
```

### 本地运行

```bash
streamlit run app.py
```

然后在浏览器中打开 http://localhost:8501

## 📁 项目结构

```
yolo_streamlit_cloud/
├── app.py                 # 主应用文件
├── turn.py                # TURN 服务器配置
├── requirements.txt       # 依赖列表
├── yolov8n.pt            # YOLOv8 模型文件
├── README.md             # 项目说明
└── WEBRTC_DEPLOYMENT_GUIDE.md  # WebRTC 部署指南
```

## 📦 依赖说明

| 依赖包 | 版本 | 说明 |
|--------|------|------|
| streamlit | >=1.28.0 | Web 应用框架 |
| streamlit-webrtc | >=0.45.0 | WebRTC 组件 |
| ultralytics | >=8.0.0 | YOLOv8 模型 |
| opencv-python-headless | >=4.8.0 | 图像处理 |
| av | >=10.0.0 | 视频帧处理 |
| twilio | >=8.1.0 | TURN 服务器（可选） |
| pandas | >=2.0.0 | 数据处理 |
| Pillow | >=10.0.0 | 图像处理 |

## ☁️ Streamlit Cloud 部署

### 步骤 1：推送代码到 GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 步骤 2：在 Streamlit Cloud 部署

1. 访问 [Streamlit Cloud](https://share.streamlit.io/)
2. 连接你的 GitHub 仓库
3. 选择 `app.py` 作为主文件
4. 点击 Deploy

### 步骤 3：配置 TURN 服务器（推荐）

在 Streamlit Cloud 的 Settings → Secrets 中添加：

```toml
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
```

#### 获取 Twilio 凭证

1. 注册 [Twilio](https://www.twilio.com/) 账号
2. 进入控制台获取 Account SID 和 Auth Token
3. Twilio 提供免费试用额度

> ⚠️ **注意**：如果不配置 TURN 服务器，应用会使用免费的 Google STUN 服务器。在某些网络环境下（如公司防火墙后）可能无法正常工作。

## 🔧 配置说明

### 置信度阈值

通过侧边栏滑块调整，范围 0.1 - 1.0：
- **低阈值 (0.1-0.3)**：检测更多对象，但可能有误检
- **中阈值 (0.4-0.6)**：平衡检测率和准确率
- **高阈值 (0.7-1.0)**：只显示高置信度检测结果

### 模型选择

当前使用 YOLOv8n（nano）模型：
- 速度最快，适合实时检测
- 支持 80 种 COCO 数据集类别

## 🐛 常见问题

### Q1: 点击 START 后摄像头无法启动？

**A:** 
- 确保浏览器已获得摄像头权限
- 使用 HTTPS 或 localhost 访问
- 尝试使用 Chrome 或 Edge 浏览器

### Q2: 部署后 WebRTC 连接失败？

**A:**
- 配置 Twilio TURN 服务器
- 检查网络是否有防火墙限制
- 查看 Streamlit Cloud 日志

### Q3: 检测速度较慢？

**A:**
- 这是正常现象，YOLOv8n 在 CPU 上运行
- 可以降低视频分辨率
- 云端部署可能比本地慢

### Q4: 统计图表不更新？

**A:**
- 确保摄像头正在运行（显示绿色状态）
- 将物体放在摄像头前以触发检测
- 统计会每 0.5 秒自动更新

## 📚 技术参考

- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) - WebRTC Streamlit 组件
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [Streamlit](https://streamlit.io/) - Web 应用框架
- [Twilio TURN](https://www.twilio.com/docs/stun-turn) - TURN 服务器文档

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

Made with ❤️ using Streamlit and YOLOv8
