import os
import tempfile
import numpy as np
import pytest
import cv2
import wave
import soundfile as sf

from quirky.plugins.dl import dl_available, require_dl, repaint, restore_face, clone_voice
from quirky.video.pipeline import VideoHumanizer


def test_dl_checks():
    if not dl_available():
        with pytest.raises(RuntimeError) as excinfo:
            require_dl()
        assert "optional neural plugin" in str(excinfo.value)


def test_repaint():
    pytest.importorskip("onnxruntime")
    
    # 64x64 test image and mask
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    # create a white spot in the middle
    image[24:40, 24:40] = 255
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    
    res = repaint(image, mask)
    assert res.shape == (64, 64, 3)
    assert res.dtype == np.uint8
    # The center should no longer be purely 255 (it should be inpainted)
    assert not np.all(res[24:40, 24:40] == 255)


def test_restore_face():
    pytest.importorskip("onnxruntime")
    
    # 128x128 face candidate
    face = np.zeros((128, 128, 3), dtype=np.uint8)
    face[32:96, 32:96] = [100, 150, 200]
    
    res = restore_face(face)
    assert res.shape == (128, 128, 3)
    assert res.dtype == np.uint8


def test_clone_voice():
    pytest.importorskip("onnxruntime")
    
    # Generate temporary source and reference wave files
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Source: 440Hz sine wave
    src_data = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32) * 0.5
    # Reference: 330Hz sine wave
    ref_data = np.sin(2.0 * np.pi * 330.0 * t).astype(np.float32) * 0.5
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_path = os.path.join(tmp_dir, "src.wav")
        ref_path = os.path.join(tmp_dir, "ref.wav")
        out_path = os.path.join(tmp_dir, "out.wav")
        
        sf.write(src_path, src_data, sr)
        sf.write(ref_path, ref_data, sr)
        
        res = clone_voice(src_path, ref_path, out_path)
        
        assert os.path.exists(out_path)
        assert res["modality"] == "voice"
        assert res["sample_rate"] == 16000
        
        # Verify output wave exists and has content
        out_data, out_sr = sf.read(out_path)
        assert out_sr == 16000
        assert len(out_data) > 0


def test_video_localized_tracking():
    # Write a dummy video file with linear motion to test tracking
    w, h = 160, 120
    frames = 10
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = os.path.join(tmp_dir, "in.mp4")
        out_path = os.path.join(tmp_dir, "out.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(in_path, fourcc, 10.0, (w, h))
        for i in range(frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Move a square rectangle
            x = 10 + i * 5
            cv2.rectangle(frame, (x, 30), (x + 20, 50), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        
        # humanize with localized bounding box tracking
        bbox = (10, 30, 20, 20)  # Initial rectangle position
        VideoHumanizer.humanize(in_path, out_path, intensity=0.5, bbox=bbox)
        
        assert os.path.exists(out_path)
        cap = cv2.VideoCapture(out_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == frames
