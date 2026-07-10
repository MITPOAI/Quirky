import numpy as np
import cv2
from typing import Tuple

class VideoHumanizer:
    @staticmethod
    def generate_fractal_drift(num_frames: int, scale: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a low-frequency fractal noise signal (smoothed random walks)
        to simulate natural handheld camera drift (dx, dy, dtheta).
        """
        # Random walks
        dx = np.cumsum(np.random.normal(0, 1, num_frames))
        dy = np.cumsum(np.random.normal(0, 1, num_frames))
        dtheta = np.cumsum(np.random.normal(0, 0.05, num_frames))
        
        # Smooth with Gaussian filter (low frequency drift)
        from scipy.ndimage import gaussian_filter1d
        dx = gaussian_filter1d(dx, sigma=15) * scale
        dy = gaussian_filter1d(dy, sigma=15) * scale
        dtheta = gaussian_filter1d(dtheta, sigma=15) * scale
        
        return dx, dy, dtheta

    @staticmethod
    def humanize(
        video_path: str,
        output_path: str,
        intensity: float = 0.5,
        t_row: float = 0.02
    ) -> None:
        """
        Applies handheld camera drift and sub-frame rolling shutter correction to video frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video at {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30.0
        if frame_count <= 0:
            frame_count = 100
            
        # Generate drift trajectory
        dx, dy, dtheta = VideoHumanizer.generate_fractal_drift(frame_count, scale=4.0 * intensity)
        
        # Setup VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            h, w, c = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Rolling Shutter Correction
            # Formula: y_corrected = y_raw + v_y(x, y) * (r/H * T_row)
            # Estimate velocity v_y using optical flow relative to previous frame
            v_y = np.zeros((h, w), dtype=np.float32)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                v_y = flow[..., 1]
                
            # Apply row-by-row shift correction
            # Map coordinates
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_y = map_y.astype(np.float32)
            map_x = map_x.astype(np.float32)
            
            # Row index fraction r/H
            r_fraction = map_y / float(h)
            
            # Apply shift
            map_y_corrected = map_y + v_y * (r_fraction * t_row * intensity * 100.0)
            map_y_corrected = np.clip(map_y_corrected, 0, h - 1)
            
            corrected_frame = cv2.remap(frame, map_x, map_y_corrected, cv2.INTER_LINEAR)
            
            # 2. Handheld Camera Drift Injection (Affine transform)
            # Center of frame
            center_x, center_y = w / 2.0, h / 2.0
            
            # Fetch current drift values
            idx = min(frame_idx, len(dx) - 1)
            cur_dx = dx[idx]
            cur_dy = dy[idx]
            cur_rot = dtheta[idx]
            
            # Compute rotation matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), cur_rot, 1.0)
            # Add translation
            M[0, 2] += cur_dx
            M[1, 2] += cur_dy
            
            # Apply warp affine
            drifted_frame = cv2.warpAffine(corrected_frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Write out
            out.write(drifted_frame)
            
            prev_gray = gray
            frame_idx += 1
            
        cap.release()
        out.release()
