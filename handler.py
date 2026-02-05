"""
RunPod Serverless Handler — Real-ESRGAN 4x + GFPGAN Face Restore + Audio Enhancement
Deploy: runpodctl deploy --handler handler.py
"""

import runpod
import subprocess
import os
import tempfile
import requests
import cv2
from pathlib import Path

REALESRGAN_MODEL = "realesr-general-x4v3"

def download_file(url: str, dest: str) -> str:
    print(f"Downloading {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest

def restore_faces_video(input_path: str, output_path: str):
    """Run GFPGAN face restoration on video frames."""
    from gfpgan import GFPGANer

    # Init GFPGAN
    restorer = GFPGANer(
        model_path='/models/GFPGANv1.4.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Restore faces
        _, _, restored = restorer.enhance(frame, paste_back=True)
        out.write(restored)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Face restoration complete: {frame_count} frames")
    return output_path

def enhance_audio(input_path: str, output_path: str):
    """Enhance audio: noise reduction + normalization."""
    # Extract audio, apply noise reduction and loudness normalization
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "highpass=f=80,lowpass=f=12000,afftdn=nf=-20,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "copy",
        output_path
    ]
    print(f"Enhancing audio...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"Audio enhance warning: {result.stderr}")
        # Don't fail, just skip audio enhancement
        return input_path
    return output_path

def run_realesrgan(input_path: str, output_path: str, scale: int = 4, model: str = REALESRGAN_MODEL):
    cmd = [
        "realesrgan-ncnn-vulkan",
        "-i", input_path,
        "-o", output_path,
        "-n", model,
        "-s", str(scale),
        "-f", "mp4"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(f"Real-ESRGAN failed: {result.stderr}")
    return output_path

def upload_result(file_path: str, upload_url: str) -> str:
    """Upload result to provided URL or return base64."""
    if upload_url:
        with open(file_path, 'rb') as f:
            resp = requests.put(upload_url, data=f, timeout=600)
            resp.raise_for_status()
        return upload_url.split('?')[0]  # Return URL without query params
    else:
        # Return file size info if no upload URL
        size = os.path.getsize(file_path)
        return f"file://{file_path} ({size} bytes)"

def handler(job):
    """
    RunPod handler for video enhancement.

    Input:
        video_url: URL to download input video
        scale: Upscale factor (2 or 4, default 4)
        model: Model name (default realesr-general-x4v3)
        face_restore: GFPGAN face enhancement (default true)
        audio_enhance: Noise reduction + normalization (default true)
        upload_url: Optional pre-signed URL to upload result
    """
    import time
    start = time.time()

    job_input = job["input"]
    video_url = job_input.get("video_url")
    scale = job_input.get("scale", 4)
    model = job_input.get("model", REALESRGAN_MODEL)
    face_restore = job_input.get("face_restore", True)
    audio_enhance = job_input.get("audio_enhance", True)
    upload_url = job_input.get("upload_url", "")

    if not video_url:
        return {"error": "video_url required"}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_ext = Path(video_url).suffix or ".mp4"
        input_path = os.path.join(tmpdir, f"input{input_ext}")
        download_file(video_url, input_path)

        # Upscale first
        upscaled_path = os.path.join(tmpdir, "upscaled.mp4")
        run_realesrgan(input_path, upscaled_path, scale, model)

        current = upscaled_path

        # Face restore after upscale
        if face_restore:
            restored_path = os.path.join(tmpdir, "restored.mp4")
            restore_faces_video(current, restored_path)
            current = restored_path

        # Audio enhancement
        if audio_enhance:
            audio_path = os.path.join(tmpdir, "audio_enhanced.mp4")
            current = enhance_audio(current, audio_path)

        result_url = upload_result(current, upload_url)

        return {
            "video_url": result_url,
            "duration_secs": round(time.time() - start, 1),
            "scale": scale,
            "face_restore": face_restore,
            "audio_enhance": audio_enhance
        }

runpod.serverless.start({"handler": handler})
