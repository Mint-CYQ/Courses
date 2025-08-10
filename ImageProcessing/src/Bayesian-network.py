import os
import numpy as np

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

class BayesianNetwork:
    def __init__(self, n_labels=10, n_pixels=784, n_values=2) -> None:
        self.n_labels = n_labels
        self.n_pixels = n_pixels
        self.n_values = n_values
        self.labels_prior = np.zeros(n_labels)
        self.pixels_cond_label = np.zeros((n_pixels, n_values, n_labels))

    def fit(self, pixels, labels):
        n_samples = len(labels)
        # 拉普拉斯平滑初始化
        self.labels_prior = np.ones(self.n_labels)
        self.pixels_cond_label = np.ones((self.n_pixels, self.n_values, self.n_labels))

        # 统计频率
        for i in range(n_samples):
            label = labels[i]
            self.labels_prior[label] += 1
            for j in range(self.n_pixels):
                self.pixels_cond_label[j, pixels[i, j], label] += 1

        # 转换成概率
        self.labels_prior /= np.sum(self.labels_prior)
        for label in range(self.n_labels):
            for j in range(self.n_pixels):
                self.pixels_cond_label[j, :, label] /= np.sum(self.pixels_cond_label[j, :, label])

        # 取对数防止下溢
        self.labels_prior = np.log(self.labels_prior)
        self.pixels_cond_label = np.log(self.pixels_cond_label)

    def predict(self, pixels):
        n_samples = pixels.shape[0]
        labels_pred = np.zeros(n_samples, dtype=np.uint8)

        for i in range(n_samples):
            log_probs = np.copy(self.labels_prior)
            for label in range(self.n_labels):
                log_probs[label] += np.sum(self.pixels_cond_label[np.arange(self.n_pixels), pixels[i], label])
            labels_pred[i] = np.argmax(log_probs)
        return labels_pred

    def score(self, pixels, labels):
        labels_pred = self.predict(pixels)
        return np.mean(labels_pred == labels)

if __name__ == '__main__':
    train_path = os.path.join(base_dir, '..', 'data', 'train.csv')
    train_path = os.path.abspath(train_path)  # 转成绝对路径
    test_path = os.path.join(base_dir, '..', 'data', 'test.csv')
    test_path = os.path.abspath(test_path)  # 转成绝对路径

    # 读取数据
    train_data = np.loadtxt(train_path, delimiter=',', dtype=np.uint8)
    test_data = np.loadtxt(test_path, delimiter=',', dtype=np.uint8)
    pixels_train, labels_train = train_data[:, :-1], train_data[:, -1]
    pixels_test, labels_test = test_data[:, :-1], test_data[:, -1]

    # 训练
    bn = BayesianNetwork()
    bn.fit(pixels_train, labels_train)

    # 测试准确率
    acc = bn.score(pixels_test, labels_test)
    print(f'Test Accuracy: {acc:.4f}')
