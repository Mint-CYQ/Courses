import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def read_image(filepath='./data/ustc-cow.png'):
    # 如果当前路径不存在，尝试上一级目录
    if not os.path.exists(filepath):
        alt_path = os.path.join(os.path.dirname(__file__), '../data/ustc-cow.png')
        alt_path = os.path.abspath(alt_path)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"找不到图片文件: {filepath} 或 {alt_path}")

    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"OpenCV 无法读取图片文件: {filepath}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class KMeans:
    def __init__(self, k=4, max_iter=20):
        self.k = k
        self.max_iter = max_iter
    
    def initialize_centers(self, points):
        n = points.shape[0]
        indices = np.random.choice(n, self.k, replace=False)
        return points[indices]
    
    def assign_points(self, centers, points):
        distances = np.linalg.norm(points[:, None] - centers[None, :], axis=2)
        return np.argmin(distances, axis=1)
    
    def update_centers(self, labels, points):
        centers = np.zeros((self.k, points.shape[1]))
        for i in range(self.k):
            cluster_points = points[labels == i]
            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)
        return centers
    
    def fit(self, points):
        centers = self.initialize_centers(points)
        for _ in range(self.max_iter):
            labels = self.assign_points(centers, points)
            new_centers = self.update_centers(labels, points)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return centers, labels
    
    def compress(self, img):
        points = img.reshape((-1, img.shape[-1]))
        centers, labels = self.fit(points)
        compressed_points = centers[labels]
        return compressed_points.reshape(img.shape)


if __name__ == '__main__':
    img = read_image('./data/ustc-cow.png')
    k_values = [2, 4, 8, 16, 32]
    for k in k_values:
        kmeans = KMeans(k=k, max_iter=20)
        compressed_img = kmeans.compress(img).astype(np.uint8)
        plt.imshow(compressed_img)
        plt.title(f'Compressed Image (k={k})')
        plt.axis('off')
        plt.savefig(f'compressed_image_k{k}.png')
        plt.show()
