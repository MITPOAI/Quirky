import numpy as np
import wave
import struct
import math
from scipy.signal import butter, lfilter

ATTRIBUTION = "Powered by Quirky (MITPO)"


def solve_lorenz(num_steps: int, dt: float = 0.01) -> np.ndarray:
    """
    Simulates the Lorenz Attractor chaotic system to get a normalized x trajectory.
    Retained as an optional 'chaotic' drift mode; the default humanizer now uses
    pink (1/f) noise, which matches human vocal micro-variation more closely.
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


def generate_pink_noise(n: int) -> np.ndarray:
    """
    Generate 1/f (pink) noise of length n, normalized to unit std. Human F0 micro-drift,
    jitter and shimmer follow a 1/f spectrum (long-range correlation), unlike white noise
    or a deterministic chaotic attractor. Vectorized via a single rFFT/irFFT pair.
    """
    if n <= 1:
        return np.zeros(max(n, 0))
    white = np.random.normal(0.0, 1.0, n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spec = spec / np.sqrt(freqs)
    pink = np.fft.irfft(spec, n=n)
    std = pink.std()
    return pink / std if std > 1e-8 else pink


def estimate_f0(segment: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> float:
    """
    Estimate a representative fundamental frequency (Hz) for a voiced segment.
    Uses librosa.yin when available (fast, accurate); falls back to autocorrelation.
    """
    try:
        import librosa
        f0 = librosa.yin(segment.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr)
        f0 = f0[np.isfinite(f0)]
        if f0.size:
            return float(np.median(f0))
    except Exception:
        pass
    # Autocorrelation fallback
    try:
        seg = segment - np.mean(segment)
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        lo = max(int(sr / fmax), 1)
        hi = min(int(sr / fmin), len(corr) - 1)
        if hi > lo:
            lag = lo + int(np.argmax(corr[lo:hi]))
            if lag > 0:
                return float(sr / lag)
    except Exception:
        pass
    return 150.0


class AudioHumanizer:
    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)
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
        if num_samples <= 1:
            return np.zeros(max(num_samples, 0))
        # White noise
        noise = np.random.normal(0.0, 1.0, num_samples)

        # Bandpass filter (breathing sound has energy concentrated in 300Hz - 2200Hz range)
        filtered = AudioHumanizer._bandpass_filter(noise, 300, 2200, sr, order=4)

        # Envelope: soft cosine attack and decay (fade-in, fade-out)
        t = np.linspace(0, 1, num_samples)
        envelope = np.sin(t * np.pi) ** 2  # Bell-like curve

        breath = filtered * envelope * amplitude
        return breath

    @staticmethod
    def apply_spectral_tilt(segment: np.ndarray, intensity: float) -> np.ndarray:
        """
        Glottal-source spectral-tilt humanization. Synthetic voices hold a static tilt;
        real vocal effort drifts (breathy <-> pressed). We crossfade between a darker
        (low-pass) and brighter (pre-emphasis) filtered copy using a slowly varying pink
        weight, then RMS-match to the input. Fully vectorized (two lfilter calls).
        """
        n = len(segment)
        if n < 8:
            return segment
        dark = lfilter([1.0], [1.0, -0.25], segment)     # gentle one-pole low-pass
        bright = lfilter([1.0, -0.25], [1.0], segment)    # mild pre-emphasis
        w = 0.5 + 0.5 * np.tanh(generate_pink_noise(n))   # slow drift centered ~0.5
        mix = (1.0 - w) * dark + w * bright
        blended = (1.0 - 0.5 * intensity) * segment + (0.5 * intensity) * mix
        r0 = np.sqrt(np.mean(segment ** 2)) + 1e-8
        r1 = np.sqrt(np.mean(blended ** 2)) + 1e-8
        return blended * (r0 / r1)

    @staticmethod
    def add_aspiration_noise(segment: np.ndarray, sr: int, intensity: float) -> np.ndarray:
        """
        Mix band-limited aspiration/breathiness noise gated to the voiced envelope,
        lowering the harmonics-to-noise ratio toward the human range (synthetic voices
        are unnaturally clean). Noise only rides where there is vocal energy.
        """
        n = len(segment)
        if n < 8:
            return segment
        noise = np.random.normal(0.0, 1.0, n)
        high = min(4000.0, sr * 0.5 - 100.0)
        filtered = AudioHumanizer._bandpass_filter(noise, 1000.0, high, sr, order=2)
        env = np.abs(segment)
        win = max(int(0.005 * sr), 1)
        kernel = np.ones(win) / win
        env_s = np.convolve(env, kernel, mode="same")
        env_s = env_s / (np.max(env_s) + 1e-8)
        return segment + filtered * env_s * (0.06 * intensity)

    @staticmethod
    def _read_audio(wav_path: str):
        """Read audio as mono float array in [-1,1]; return (audio, samplerate)."""
        try:
            import soundfile as sf
            data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                data = data[:, 0]
            return data.astype(float), int(sr)
        except Exception:
            pass
        # wave fallback (PCM only)
        with wave.open(wav_path, 'rb') as wav:
            params = wav.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            raw_data = wav.readframes(nframes)
        if sampwidth == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16).astype(float) / 32768.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(float) - 128.0) / 128.0
        else:
            raise ValueError("Unsupported sample width")
        if nchannels > 1:
            audio = audio[0::nchannels]
        return audio, framerate

    @staticmethod
    def _write_audio(output_path: str, audio: np.ndarray, sr: int) -> None:
        audio = np.clip(audio, -1.0, 1.0)
        try:
            import soundfile as sf
            sf.write(output_path, audio.astype(np.float32), sr, subtype="PCM_16")
            return
        except Exception:
            pass
        out_i16 = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
        with wave.open(output_path, 'wb') as out_wav:
            out_wav.setparams((1, 2, sr, len(out_i16), 'NONE', 'not compressed'))
            out_wav.writeframes(out_i16.tobytes())

    @staticmethod
    def humanize(
        wav_path: str,
        output_path: str,
        intensity: float = 0.5,
        breath_freq: float = 0.8
    ) -> dict:
        """
        Applies speech activity partitioning (VAD), pitch-period jitter/shimmer at human
        magnitudes, pink-noise micro-prosody drift, drifting glottal spectral tilt,
        aspiration noise, and procedural breath overlays.
        """
        audio, framerate = AudioHumanizer._read_audio(wav_path)

        # 1. Voice Activity Detection (VAD) via short-time energy
        frame_size = max(int(0.02 * framerate), 1)  # 20ms frames
        hop_size = max(int(0.01 * framerate), 1)    # 10ms hop

        num_frames = max((len(audio) - frame_size) // hop_size + 1, 0)
        energies = []
        for i in range(num_frames):
            frame = audio[i * hop_size: i * hop_size + frame_size]
            energies.append(np.mean(frame ** 2))

        energies = np.array(energies) if energies else np.array([0.0])
        threshold = np.mean(energies) * 0.15
        is_speech = energies > threshold

        # Smooth is_speech decisions (fill small silence gaps)
        smoothed_speech = np.copy(is_speech)
        for i in range(2, len(is_speech) - 2):
            if is_speech[i - 2] and is_speech[i + 2]:
                smoothed_speech[i] = True

        # Find voiced intervals
        voiced_intervals = []
        in_speech = False
        start_idx = 0
        for i, val in enumerate(smoothed_speech):
            if val and not in_speech:
                start_idx = i * hop_size
                in_speech = True
            elif not val and in_speech:
                end_idx = i * hop_size + frame_size
                voiced_intervals.append((start_idx, end_idx))
                in_speech = False
        if in_speech:
            voiced_intervals.append((start_idx, len(audio)))

        # If nothing was detected as speech, treat the whole clip as one voiced span
        if not voiced_intervals and len(audio) > 0:
            voiced_intervals = [(0, len(audio))]

        # 2. Assemble output with breath insertion + per-period jitter/shimmer
        output_buffer = []
        last_end = 0

        for idx, (start, end) in enumerate(voiced_intervals):
            silence_len = start - last_end

            # Insert a procedural breath into long enough pauses
            if silence_len > int(0.2 * framerate) and np.random.rand() < breath_freq:
                pre_silence = int(0.05 * framerate)
                post_silence = int(0.05 * framerate)
                breath_len_samples = silence_len - pre_silence - post_silence
                breath_len_sec = min(float(breath_len_samples) / framerate, 0.5)
                breath_samples = AudioHumanizer.generate_procedural_breath(
                    breath_len_sec, framerate, amplitude=0.03 * intensity
                )
                output_buffer.append(audio[last_end: last_end + pre_silence])
                output_buffer.append(breath_samples)
                output_buffer.append(audio[last_end + pre_silence + len(breath_samples): start])
            else:
                output_buffer.append(audio[last_end: start])

            # Micro-pause: sometimes let the delivery breathe -- insert 80-250ms of
            # low-level room tone (not digital silence) before a phrase, the hesitation
            # rhythm TTS never produces. Slightly extends duration by design.
            if idx > 0 and np.random.rand() < 0.3 * intensity:
                pause_len = int(np.random.uniform(0.08, 0.25) * framerate)
                room = np.random.normal(0.0, 1.0, pause_len)
                room = AudioHumanizer._bandpass_filter(room, 80.0, 1200.0, framerate, order=2)
                output_buffer.append(room * 0.0015)

            # Pitch-period jitter/shimmer + tilt + aspiration on the voiced segment
            segment = audio[start:end]
            seg_len = len(segment)
            if seg_len > 8:
                f0 = estimate_f0(segment, framerate)
                period = max(int(framerate / max(f0, 1e-3)), 1)
                n_periods = max(seg_len // period, 4)

                # Jitter: per-period fractional period-length variation (~0.3-1.0% * intensity)
                jitter_mag = 0.008 * intensity
                jitter_seq = generate_pink_noise(n_periods) * jitter_mag
                anchors = np.linspace(0, seg_len - 1, n_periods)
                speed_curve = 1.0 + np.interp(np.arange(seg_len), anchors, jitter_seq)

                # Intonation: F0 declination -- human phrases start slightly higher-pitched
                # and drift down. A slow speed ramp (fast->slow) shifts pitch the same way
                # since the warp is plain resampling. ~2% end-to-end at full intensity.
                decl = 0.02 * intensity
                speed_curve *= np.linspace(1.0 + decl, 1.0 - decl, seg_len)

                # Phrase-final lengthening: humans stretch the last ~200ms of a phrase.
                tail = min(int(0.2 * framerate), seg_len // 3)
                if tail > 8:
                    speed_curve[-tail:] *= (1.0 - 0.06 * intensity)

                coords = np.cumsum(speed_curve)
                coords = coords / coords[-1] * (seg_len - 1)
                warped = np.interp(coords, np.arange(seg_len), segment)

                # Shimmer: per-period amplitude variation (~3-5% * intensity)
                shimmer_mag = 0.04 * intensity
                shimmer_seq = generate_pink_noise(n_periods) * shimmer_mag
                shimmer_curve = 1.0 + np.interp(np.arange(seg_len), anchors, shimmer_seq)
                warped = warped * shimmer_curve

                # Glottal spectral-tilt drift + aspiration/breathiness
                warped = AudioHumanizer.apply_spectral_tilt(warped, intensity)
                warped = AudioHumanizer.add_aspiration_noise(warped, framerate, intensity)

                output_buffer.append(warped)
            elif seg_len > 0:
                output_buffer.append(segment)

            last_end = end

        if last_end < len(audio):
            output_buffer.append(audio[last_end:])

        out_audio = np.concatenate(output_buffer) if output_buffer else audio
        AudioHumanizer._write_audio(output_path, out_audio, framerate)

        return {
            "attribution": ATTRIBUTION,
            "modality": "audio",
            "params": {"intensity": intensity, "breath_freq": breath_freq},
            "sample_rate": framerate,
            "output_path": output_path,
        }
