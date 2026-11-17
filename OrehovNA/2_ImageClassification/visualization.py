# visualization.py
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # не требует GUI, работает на серверах/в ноутбуках без дисплея
import matplotlib.pyplot as plt


def plot_histogram(image, histogram, save_path=None, title=None):
    """Совместный вывод: изображение + его BoW-гистограмма."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title or 'Image')
    axes[0].axis('off')

    axes[1].bar(np.arange(len(histogram)), histogram, width=1.0)
    axes[1].set_title('Bag-of-Words histogram')
    axes[1].set_xlabel('Visual word index')
    axes[1].set_ylabel('Frequency (L2-normalized)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def draw_keypoints(image, extractor, save_path=None):
    """Полезно для отладки SIFT/ORB: показывает, где нашлись признаки."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(extractor, 'sift'):
        kp = extractor.sift.detect(gray, None)
    elif hasattr(extractor, 'orb'):
        kp = extractor.orb.detect(gray, None)
    else:
        return None
    vis = cv2.drawKeypoints(image, kp, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if save_path:
        cv2.imwrite(save_path, vis)
    return vis