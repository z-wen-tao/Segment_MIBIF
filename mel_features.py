# 梅尔滤波器
import numpy as np

def frame(data, window_length, hop_length):
    num_samples = data.shape[0]  # 总样本数
    num_frames = int(np.floor((num_samples - window_length)/hop_length))  # 总帧数
    # print(num_frames)
    shape = (num_frames, window_length) + data.shape[1:]  # (帧数量,帧长度)
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def hertz_to_mel(frequencies_hertz):

    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
    )

def split_frame(signal, fft_length, hop_length=None, windows_length=None):
    frames = frame(signal, windows_length, hop_length)  # 分帧
    window = 0.5-(0.5 * np.cos(2 * np.pi / windows_length * np.arange(windows_length)))  # 生成一个汉宁窗口
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))  # fft=128

def spectrogram_to_mel_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=129,
        audio_sample_rate=8000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=3800.0,
):
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            "lower_edge_hertz %.1f >= upper_edge_hertz %.1f"
            % (lower_edge_hertz, upper_edge_hertz)
        )
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError(
            "upper_edge_hertz %.1f is greater than Nyquist %.1f"
            % (upper_edge_hertz, nyquist_hertz)
        )
    #print(nyquist_hertz)
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2
    )
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i: i + 3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (
                center_mel - lower_edge_mel
        )
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (
                upper_edge_mel - center_mel
        )
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix

def log_mel_spectrogram(
        data,
        audio_sample_rate,
        log_offset=0.0,
        window_length_secs=0.05,
        hop_length_secs=0.020,
        **kwargs
):
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))  # 算窗口长度和跳跃长度的样本数
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples)/np.log(2.0)))  # 将FFT的长度设置为大于等于窗口长度的最小2的幂次方，以便在进行FFT计算时能够高效地处理信号数据
    # 分帧
    spectrogram = split_frame(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        windows_length=window_length_samples
    )
    mel_spectrogram = np.dot(
        spectrogram,
        spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1],
            audio_sample_rate=audio_sample_rate,
            **kwargs
        ),
    )
    return np.log(mel_spectrogram+log_offset)
