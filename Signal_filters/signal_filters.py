import numpy as np
import matplotlib.pyplot as plt
import os


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def custom_FFT(x):
    N = len(x)
    if N <= 1:
        return x
    even = custom_FFT(x[0::2])
    odd = custom_FFT(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [
        even[k] - T[k] for k in range(N // 2)
    ]


def custom_IFFT(X):
    N = len(X)
    X = [x.conjugate() for x in X]
    x = custom_FFT(X)
    x = [xi.conjugate() / N for xi in x]
    return x


def fftshift(X):
    N = len(X)
    return np.roll(X, N // 2)


def generate_signal(t, freqs):
    signal = np.zeros_like(t)
    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * t)
    return signal


def fft(signal):
    n = next_power_of_2(len(signal))
    padded_signal = np.pad(signal, (0, n - len(signal)))
    return fftshift(custom_FFT(padded_signal))


def lowpass_filter(signal, cutoff_freq, sampling_rate):
    spectrum = fft(signal)
    frequencies = np.linspace(-sampling_rate / 2, sampling_rate / 2, len(spectrum))
    filtered_spectrum = spectrum * (np.abs(frequencies) <= cutoff_freq)
    return np.real(custom_IFFT(fftshift(filtered_spectrum)))


def highpass_filter(signal, cutoff_freq, sampling_rate):
    spectrum = fft(signal)
    frequencies = np.linspace(-sampling_rate / 2, sampling_rate / 2, len(spectrum))
    filtered_spectrum = spectrum * (np.abs(frequencies) >= cutoff_freq)
    return np.real(custom_IFFT(fftshift(filtered_spectrum)))


def mse(original_signal, filtered_signal):
    return np.mean((original_signal - filtered_signal) ** 2)


def main():
    signals_dir = "signals"
    if not os.path.exists(signals_dir):
        os.makedirs(signals_dir)

    sampling_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = generate_signal(t, [5, 10, 20, 30, 40])

    lowpass_signal = lowpass_filter(signal, 15, sampling_rate)
    highpass_signal = highpass_filter(signal, 25, sampling_rate)

    lowpass_mse = mse(signal[: len(t)], lowpass_signal[: len(t)])
    highpass_mse = mse(signal[: len(t)], highpass_signal[: len(t)])

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, signal)
    plt.title("Original Signal")

    plt.subplot(3, 1, 2)
    plt.plot(t, lowpass_signal[: len(t)])
    plt.title(f"Signal after Lowpass Filter (MSE: {lowpass_mse:.2f})")

    plt.subplot(3, 1, 3)
    plt.plot(t, highpass_signal[: len(t)])
    plt.title(f"Signal after Highpass Filter (MSE: {highpass_mse:.2f})")

    plt.tight_layout()
    plt.savefig(os.path.join(signals_dir, "signals.png"))
    plt.show()


if __name__ == "__main__":
    main()
