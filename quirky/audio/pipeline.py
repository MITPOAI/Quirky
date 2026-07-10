import numpy as np
import wave
import struct
import math
from scipy.signal import butter, lfilter

def solve_lorenz(num_steps: int, dt: float = 0.01) -> np.ndarray:
    """
    Simulates the Lorenz Attractor chaotic system to get a normalized x trajectory.
    """
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    x = 0.1
    y = 0.0
    z = 0.0
    
    xs = np.zeros(num_steps)
    for i in range(num_steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        xs[i] = x
        
    xs_min, xs_max = np.min(xs), np.max(xs)
    if xs_max > xs_min:
        xs = 2.0 * (xs - xs_min) / (xs_max - xs_min) - 1.0
    else:
        xs = np.zeros_like(xs)
    return xs

class AudioHumanizer:
    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def _bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = AudioHumanizer._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def generate_procedural_breath(duration_sec: float, sr: int, amplitude: float = 0.05) -> np.ndarray:
        """
        Procedurally synthesizes a natural human breath (inhalation) using
        envelope-shaped bandpass-filtered white noise.
        """
        num_samples = int(duration_sec * sr)
        # White noise
        noise = np.random.normal(0.0, 1.0, num_samples)
        
        # Bandpass filter (breathing sound has energy concentrated in 300Hz - 2500Hz range)
        filtered = AudioHumanizer._bandpass_filter(noise, 300, 2200, sr, order=4)
        
        # Envelope: soft cosine attack and decay (fade-in, fade-out)
        t = np.linspace(0, 1, num_samples)
        envelope = np.sin(t * np.pi) ** 2  # Bell-like curve
        
        breath = filtered * envelope * amplitude
        return breath

    @staticmethod
    def humanize(
        wav_path: str,
        output_path: str,
        intensity: float = 0.5,
        breath_freq: float = 0.8
    ) -> None:
        """
        Applies speech activity partitioning (VAD), prosody speed-stretching, and procedural breath overlays.
        """
        # Read wav
        with wave.open(wav_path, 'rb') as wav:
            params = wav.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            raw_data = wav.readframes(nframes)
            
        # Convert to numpy float array normalized to [-1.0, 1.0]
        if sampwidth == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(float) / 32768.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(float) - 128.0) / 128.0
        else:
            raise ValueError("Unsupported sample width")
            
        # For simplicity, if multi-channel, process channel 0
        if nchannels > 1:
            audio = audio[0::nchannels]
            
        # 1. Voice Activity Detection (VAD)
        # Calculate short-time energy
        frame_size = int(0.02 * framerate) # 20ms frames
        hop_size = int(0.01 * framerate)   # 10ms hop
        
        num_frames = (len(audio) - frame_size) // hop_size + 1
        energies = []
        for i in range(num_frames):
            frame = audio[i*hop_size : i*hop_size + frame_size]
            energies.append(np.mean(frame**2))
            
        energies = np.array(energies)
        # Threshold for speech presence (VAD)
        threshold = np.mean(energies) * 0.15
        is_speech = energies > threshold
        
        # Smooth is_speech decisions
        smoothed_speech = np.copy(is_speech)
        for i in range(2, len(is_speech)-2):
            # Fill small silence gaps (speech smoothing)
            if is_speech[i-2] and is_speech[i+2]:
                smoothed_speech[i] = True
                
        # Find boundaries of voiced segments and pauses
        voiced_intervals = []
        in_speech = False
        start_idx = 0
        
        for i, val in enumerate(smoothed_speech):
            if val and not in_speech:
                # Speech started
                start_idx = i * hop_size
                in_speech = True
            elif not val and in_speech:
                # Speech ended
                end_idx = i * hop_size + frame_size
                voiced_intervals.append((start_idx, end_idx))
                in_speech = False
                
        if in_speech:
            voiced_intervals.append((start_idx, len(audio)))
            
        # 2. Apply Speed Stretch Jitter and Assemble Output
        output_buffer = []
        last_end = 0
        
        for idx, (start, end) in enumerate(voiced_intervals):
            # Silence/pause duration
            silence_len = start - last_end
            
            # If silence is long enough (e.g. > 200ms), insert a procedural breath
            if silence_len > int(0.2 * framerate) and np.random.rand() < breath_freq:
                # Keep some silence before and after breath
                pre_silence = int(0.05 * framerate) # 50ms pause
                post_silence = int(0.05 * framerate) # 50ms pause
                breath_len_samples = silence_len - pre_silence - post_silence
                breath_len_sec = float(breath_len_samples) / framerate
                
                # Limit breath to max 0.5s
                breath_len_sec = min(breath_len_sec, 0.5)
                breath_samples = AudioHumanizer.generate_procedural_breath(breath_len_sec, framerate, amplitude=0.03 * intensity)
                
                # Append pre-silence
                output_buffer.append(audio[last_end : last_end + pre_silence])
                # Append breath
                output_buffer.append(breath_samples)
                # Append remainder of silence
                output_buffer.append(audio[last_end + pre_silence + len(breath_samples) : start])
            else:
                # Just append original silence
                output_buffer.append(audio[last_end : start])
                
            # Chaotic Vocal Micro-instability (Jitter and Shimmer using Lorenz Attractor)
            segment = audio[start:end]
            seg_len = len(segment)
            if seg_len > 0:
                # Generate Lorenz attractor chaotic trajectory
                xs = solve_lorenz(seg_len, dt=0.005)
                
                # Jitter: scale chaotic variable x(t) to bounds of +-0.02% * intensity
                jitter_factor = 0.0002 * intensity
                speed_curve = 1.0 + xs * jitter_factor
                
                # Integrate speed curve to compute dynamic time warp coordinates
                coords = np.cumsum(speed_curve)
                coords = coords / coords[-1] * (seg_len - 1)
                
                # Apply time warp (Jitter)
                stretched_segment = np.interp(coords, np.arange(seg_len), segment)
                
                # Shimmer: modulate amplitude by +-0.02% * intensity using chaotic attractor path
                shimmer_factor = 0.0002 * intensity
                shimmer_multiplier = 1.0 + xs * shimmer_factor
                stretched_segment = stretched_segment * shimmer_multiplier
                
                output_buffer.append(stretched_segment)
            
            last_end = end
            
        # Append trailing audio
        if last_end < len(audio):
            output_buffer.append(audio[last_end:])
            
        # Concatenate and write wav
        out_audio = np.concatenate(output_buffer)
        out_audio = np.clip(out_audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        
        with wave.open(output_path, 'wb') as out_wav:
            # Output mono wav
            out_wav.setparams((1, 2, framerate, len(out_audio), 'NONE', 'not compressed'))
            out_wav.writeframes(out_audio.tobytes())
