import numpy as np

def attack_drift(x, frac=0.05, strength=2.5):
    """
    Drift تدريجي: نختار مقطع من البيانات ونضيف انحراف يزيد تدريجياً.
    """
    x = x.copy()
    n, d = x.shape
    L = max(5, int(n * frac))
    start = np.random.randint(0, n - L)
    drift = np.linspace(0, strength, L).reshape(-1, 1)
    x[start:start+L] += drift
    y = np.zeros(n, dtype=np.uint8)
    y[start:start+L] = 1
    return x, y

def attack_high_noise(x, frac=0.05, sigma=0.8):
    """
    Noise عالي: نضيف Gaussian noise قوي على مقطع.
    """
    x = x.copy()
    n, d = x.shape
    L = max(5, int(n * frac))
    start = np.random.randint(0, n - L)
    noise = np.random.normal(0, sigma, size=(L, d))
    x[start:start+L] += noise
    y = np.zeros(n, dtype=np.uint8)
    y[start:start+L] = 1
    return x, y

def attack_channel_freeze(x, frac=0.05, channels=None):
    """
    Channel Freeze: نخلي قنوات معينة تثبت على قيمة واحدة في مقطع.
    """
    x = x.copy()
    n, d = x.shape
    L = max(5, int(n * frac))
    start = np.random.randint(0, n - L)

    if channels is None:
        k = max(1, d // 10)  # 10% من القنوات
        channels = np.random.choice(d, size=k, replace=False)

    freeze_val = x[start, channels]  # قيمة ثابتة
    x[start:start+L, channels] = freeze_val

    y = np.zeros(n, dtype=np.uint8)
    y[start:start+L] = 1
    return x, y

def attack_pattern_shift(x, frac=0.05, shift=20):
    """
    Pattern Shift: نعمل circular shift للميزات (feature order) في مقطع.
    """
    x = x.copy()
    n, d = x.shape
    L = max(5, int(n * frac))
    start = np.random.randint(0, n - L)

    x[start:start+L] = np.roll(x[start:start+L], shift=shift, axis=1)

    y = np.zeros(n, dtype=np.uint8)
    y[start:start+L] = 1
    return x, y