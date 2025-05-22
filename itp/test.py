import numpy as np

def sinc_interp(x, t, t_new, M=4, window='hamming'):
    """
    Interpolate signal x sampled at times t to new times t_new.
    x: input samples
    t: sample times
    t_new: times to interpolate
    M: half-width of window
    window: window function type
    """
    T = t[1] - t[0]  # Assume uniform sampling
    x_new = np.zeros(len(t_new))
    
    for i, tn in enumerate(t_new):
        # Find samples within window
        n_start = int(np.floor(tn/T - M))
        n_end = int(np.floor(tn/T + M))
        sum_val = 0
        
        for n in range(max(0, n_start), min(len(x), n_end + 1)):
            tau = tn - n * T
            # Sinc function
            sinc_val = np.sinc(tau / T)  # np.sinc is sin(pi*x)/(pi*x)
            # Window function
            if window == 'hamming':
                if abs(tau) <= M * T:
                    w = 0.54 + 0.46 * np.cos(np.pi * tau / (M * T))
                else:
                    w = 0
            else:  # Rectangular window
                w = 1 if abs(tau) <= M * T else 0
            # Kernel
            h = sinc_val * w
            sum_val += x[n] * h
        
        x_new[i] = sum_val
    
    return x_new

# Example usage
t = np.arange(0, 10, 1)  # Sample times
x = np.sin(0.5 * t)      # Sampled signal
t_new = np.arange(0, 9.5, 0.1)  # Finer time grid
x_interp = sinc_interp(x, t, t_new, M=4, window='hamming')

# Optional: Plot results
import matplotlib.pyplot as plt
plt.plot(t, x, 'o', label='Samples')
plt.plot(t_new, x_interp, '-', label='Interpolated')
plt.legend()
plt.show()