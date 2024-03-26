import matplotlib.pyplot as plt
import numpy as np


def n_round(x: np.ndarray, n: int = 5):
    y = np.zeros_like(x)
    for i in range(1, x.shape[-1]):
        y[..., i] = np.mean(x[..., max(i - n, 0): i])
    return y


def draw_filled_curl(y: np.ndarray, x: np.ndarray = None):
    if x is None:
        x = np.arange(y.shape[0])

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', label='Curve')
    plt.fill_between(x, y, color='lightblue', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('title')
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_histogram(count_dict: dict, x_label='X', y_label='Y', max_y=None, title=None):
    labels = []
    counts = []
    for item in count_dict.items():
        labels.append(item[0])
        counts.append(item[1])
    max_y_lim = max(counts) if max_y is None else max_y
    plt.figure()
    plt.bar(labels, counts, width=0.7, align='center')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, max_y_lim * 1.1)
    plt.show()
