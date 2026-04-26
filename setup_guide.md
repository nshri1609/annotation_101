# VideoAnnotator Setup & Usage Guide

This guide covers the complete setup of the VideoAnnotator application and how to use the batch processing script to annotate videos.

## 📋 Prerequisites

- **Python**: Version 3.12 (Mandatory)
- **uv**: Modern Python package manager
- **FFmpeg**: Required for video processing

### Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 🛠️ Initial Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/InfantLab/VideoAnnotator.git
   cd VideoAnnotator
   ```

2. **Sync Dependencies**:
   This will create a virtual environment and install all necessary packages including PyTorch, Ultralytics, and MediaPipe.
   ```bash
   uv sync
   ```

3. **Initialize the Database**:
   This setup is required for the API server and job management.
   ```bash
   uv run videoannotator setup-db --admin-email admin@example.com --admin-username admin
   ```
   *Note: Save the API key printed at the end of this command.*

## 📹 Annotating Videos (Batch Mode)

The most direct way to annotate videos with visual overlays (hands and objects) is using the `batch_annotate.py` script.

### 1. Prepare your videos
Create a folder named `input_videos` in the project root and place your video files (`.mp4`, `.mov`, or `.avi`) inside it.

```bash
mkdir -p input_videos
# Move your videos into input_videos/
```

### 2. Run the Annotation Script
Run the following command to process all videos in the `input_videos` folder.

```bash
uv run python batch_annotate.py --hands --objects
```

#### Command Options:
- `--hands`: Enables hand skeleton tracking (MediaPipe).
- `--objects`: Enables object detection and tracking (YOLO11).
- `--input-dir`: (Optional) Change the source folder (default: `input_videos`).
- `--output-dir`: (Optional) Change the results folder (default: `annotated_videos`).

### 3. View Results
All annotated videos with rendered overlays will be saved in the `annotated_videos/` directory.

---

## 🌐 Running as an API Server

If you prefer to use the REST API for processing:

1. **Start the Server**:
   ```bash
   uv run videoannotator server --host 127.0.0.1 --port 18011
   ```

2. **Submit a Job** (via `curl`):
   ```bash
   curl -X POST "http://127.0.0.1:18011/api/v1/jobs/" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -F "video=@input_videos/your_video.mp4" \
     -F "selected_pipelines=person_tracking,object_detection"
   ```

3. **Monitor Jobs**:
   Visit [http://127.0.0.1:18011/docs](http://127.0.0.1:18011/docs) for the interactive API documentation.

## 📦 Requirements Summary
The application relies on several high-performance libraries:
- **Core**: FastAPI, SQLAlchemy, Pydantic
- **Computer Vision**: Ultralytics (YOLO11), MediaPipe, OpenCV
- **Audio**: OpenAI Whisper, pyannote.audio
- **Machine Learning**: PyTorch, Transformers
