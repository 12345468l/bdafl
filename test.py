import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from prettytable import PrettyTable
import matplotlib
import random
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def get_testloader():
    means = (0.4914, 0.4822, 0.4465)
    lvars = (0.2023, 0.1994, 0.2010)

    normalize = transforms.Normalize(means, lvars)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='.data/',
        train=False,
        download=True,
        transform=transform_test)
    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False, num_workers=0)
    return test_loader


def get_data():
    x0 = [7551.2637, 5211.4966, 4743.9937, 5015.2871, 11346.4512, 8546.9082,
          13023.2559, 4090.6978, 8343.8887, 12375.0576, 6893.1323, 2218.0076,
          10345.5146, 14948.2246, 6512.1221, 4765.2285, 5516.0327, 7695.8706,
          4489.2144, 10297.7148, 8246.1631, 3273.8096, 2844.9473, 19526.5586,
          9805.7285, 20425.6621, 1656.1715, 7144.1567, 12641.4053, 7712.4502,
          13106.6602, 8811.7041, 6241.5771, 5391.3379, 15518.0566, 11443.8818,
          3259.0540, 10909.6113, 1030.8605, 11444.2686, 2293.1343, 10355.8418,
          3943.5059, 4912.4644, 6793.2119, 9551.3018, 10804.6338, 7748.6118,
          10119.4873, 1066.9818, 7837.1919, 4555.5596, 8870.5098, 12306.7227,
          449.5393, 15464.4307, 11524.6201, 3438.8494, 2868.1694, 3207.0659,
          3623.3838, 14742.8721, 2784.8638, 7140.7144]
    x2 = [7116.0532, 4786.7100, 5293.2661, 6026.4395, 11229.4678, 8266.9316,
          12283.3262, 4855.8804, 9657.7959, 12539.6211, 7245.6367, 2421.6375,
          10040.2139, 13418.0781, 7705.6777, 4830.0742, 4433.1694, 8850.3311,
          5033.6060, 11294.4170, 8536.8389, 2929.1875, 3272.8345, 19346.6172,
          8059.4150, 20024.1680, 1484.4337, 7126.3286, 11389.6514, 9243.1221,
          13152.8027, 8840.0488, 6134.8018, 6393.7515, 14606.5361, 12921.7295,
          4336.8184, 10180.1865, 874.1537, 13476.7188, 1497.6798, 10578.4170,
          3694.8723, 4400.6445, 7012.9565, 8463.2168, 10014.4756, 8902.9561,
          10334.5049, 972.2934, 7472.7397, 3965.0032, 8226.2510, 12993.4961,
          251.8097, 16945.5391, 7602.7466, 5004.6958, 2131.9561, 3496.2830,
          2582.2559, 14057.0967, 4915.9170, 8449.1064]
    x3 = [7506.9907, 4985.2251, 5677.3540, 5205.5601, 11162.9209, 7576.6221,
          9966.7266, 3496.5574, 8353.9961, 10742.1172, 7287.6348, 2455.8406,
          10888.1738, 13897.2998, 7778.1260, 5490.5005, 4672.6270, 8210.6401,
          5851.2080, 10415.8408, 8698.8818, 2828.9917, 3792.3428, 16526.4873,
          7802.8569, 18744.3086, 1417.8666, 5315.7559, 11196.0947, 7505.4497,
          11440.4883, 8025.6089, 6667.5332, 5773.2407, 14439.3027, 10825.3271,
          4702.0405, 10654.9346, 860.1422, 11930.5781, 2763.1875, 8838.7070,
          3402.9446, 3423.9651, 6546.9995, 9050.1885, 9725.2139, 8673.4229,
          8324.9995, 1416.9159, 7902.6235, 3803.7568, 8530.8887, 11971.8047,
          549.0416, 15274.2549, 8139.3530, 4552.0566, 2567.8445, 3489.8591,
          3030.7537, 14396.2715, 3980.7986, 7869.4272]

    x1 = [5960.7700, 3218.4351, 2783.0156, 4742.6548, 9778.2510, 4646.0044,
          11323.2988, 4407.3843, 7279.3784, 7403.2485, 4387.7573, 2719.8804,
          5348.4717, 11685.7305, 5341.5151, 4480.7910, 4548.9858, 10162.4922,
          4038.0994, 10804.2676, 6389.3872, 3561.0928, 3422.6794, 16473.6543,
          7614.6431, 15818.6670, 1733.9124, 5605.4102, 7741.3413, 7211.7837,
          10927.3193, 9383.2959, 4763.6143, 7158.6958, 16890.3438, 7223.1533,
          3637.2720, 9031.3496, 1939.6688, 10965.8057, 2289.9973, 7755.1665,
          1656.8635, 2593.3831, 6869.1704, 10106.1963, 9346.1641, 5540.6772,
          11281.5020, 1284.0010, 7704.1167, 2582.3438, 8448.1084, 8850.3486,
          1185.8521, 15951.6582, 7338.4150, 4051.0920, 2819.4702, 2449.2935,
          2580.8123, 7846.5752, 4283.4897, 8749.5938]
    x4 = [5927.4917, 2378.3147, 3490.6021, 5360.6499, 9869.8770, 4719.7183,
          10006.1787, 4765.0415, 7983.8164, 8867.9824, 5460.7378, 2244.7410,
          5542.4209, 12195.0791, 6422.0986, 4114.4087, 3890.6099, 9885.8740,
          5191.4092, 11279.1641, 5556.1094, 5303.2534, 2102.2844, 15287.8857,
          8896.0732, 16913.5508, 1231.9818, 7167.5649, 10651.8174, 7799.8027,
          9827.7197, 10319.1396, 3896.1196, 7815.4966, 16708.4492, 6926.9668,
          2762.2297, 7539.8672, 580.3586, 9686.1445, 1935.8552, 7002.6382,
          3660.2874, 3806.0464, 7897.6606, 8861.3154, 7628.0303, 6087.7354,
          12341.9141, 1156.2924, 8095.1885, 2990.4839, 8212.5859, 8430.4092,
          164.2392, 14619.2930, 7513.8398, 4165.4976, 2753.8472, 3061.4526,
          2048.6545, 8737.9414, 5266.6997, 8884.1270]
    x5 = [5576.0830, 3585.2959, 3068.7080, 5234.7471, 10488.9795, 5281.6724,
          10803.7568, 3884.7717, 6349.6548, 7990.4702, 5282.5269, 2346.3108,
          5136.5591, 12071.2881, 6343.0708, 3771.5615, 4340.4180, 9213.5605,
          4681.8633, 11626.6416, 4333.9546, 4106.0859, 3677.0747, 15643.0293,
          7483.3506, 17006.0723, 1494.8053, 5334.5693, 9686.0039, 7781.2920,
          9472.2188, 9426.4014, 4469.4048, 8384.2021, 18259.6074, 7379.2920,
          4035.9192, 8209.1592, 790.1790, 11905.2500, 2331.9944, 8949.0859,
          2752.1626, 2647.6826, 8825.6709, 9340.7480, 7936.9688, 6642.4917,
          12537.1172, 1243.4119, 8280.0498, 4130.3203, 7790.8193, 8433.5732,
          827.6620, 16126.6484, 6882.4097, 3790.1389, 3303.5535, 2832.1096,
          3136.2881, 8334.9873, 4686.8096, 7159.1187]
    x = np.asarray([x0, x1, x2, x3, x4, x5])
    return x


def apply_standard_scaler(gradients):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    return scaler.fit_transform(gradients)


def calculate_pca_of_gradients(gradients, num_components):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components)

    # logger.info("Computing {}-component PCA of gradients".format(num_components))

    return pca.fit_transform(gradients)


def plot_gradients_2d(gradients, POISONED_WORKER_IDS):
    id = list(range(50))
    fig = plt.figure()
    SAVE_NAME = "defense_results.jpg"
    SAVE_SIZE = (18, 14)
    for (worker_id, gradient) in zip(id, gradients):
        if worker_id in POISONED_WORKER_IDS:
            plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
        else:
            plt.scatter(gradient[0], gradient[1], color="orange", s=180)

    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(False)
    plt.margins(0, 0)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)


def plot_res():
    x = load_np()
    y0 = x[0]
    y1 = x[1]
    x1 = np.linspace(0, 600, 50)
    with plt.style.context(['science', 'ieee', 'grid', 'no-latex']):
        #     matplotlib.get_cachedir()
        #     matplotlib.rc('font', family='times new roman')
        fig, ax = plt.subplots(figsize=(2.5, 1.875))
        #     fig, ax = plt.figure(figsize=(6, 6.5))

        #     for p in [10, 20, 50]:
        #     , marker='o', markersize=2

        ax.plot(x1, y0, label='FedAvg', linestyle='--', color='g')
        ax.plot(x1, y1, label='FedAsync', linestyle='--', color='dodgerblue')
        # ax.plot(x1, y2, label='WKAFL', linestyle='--', color='orange')
        # ax.plot(x1, y3, label='ours', linestyle=None, color='r')
        ax.legend()

        #     ax.set_ylim(0, 0.55)
        #     my_y_ticks = np.arange(0, 0.55,0.1)
        #     ax.set_yticks(my_y_ticks)

        ax.set(xlabel='Wall Clock Time(s)')
        ax.set(ylabel='Test Accuracy')
        #     ax.autoscale(tight=True)
        #     fig.savefig('cifar_conver_N.png', dpi=300)
        plt.show()

def load_np():
    loaded_arrays = []
    with open('res.npy', 'rb') as f:
        while True:
            try:
                loaded_array = np.load(f)
                loaded_arrays.append(loaded_array)
            except:
                break

                # 打印加载的数组
    # for array in loaded_arrays:
    # print(array.shape)
    print(len(loaded_arrays))
    return np.asarray(loaded_arrays)


def get_means():
    data = get_data()
    mean = np.sum(data, axis=0) / 6
    for i in range(6):
        data[i] -= mean
    return data


def k_means():
    import numpy as np
    from sklearn.cluster import KMeans

    # 假设你的矩阵名为 data
    data = get_data()
    print(data.shape)
    # data = apply_standard_scaler(data)
    # 将矩阵转换为一维数组
    data_flattened = data.flatten()
    print(data_flattened.shape)
    # 聚类数
    k = 2

    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    cluster_assignments = kmeans.fit_predict(data)

    # 绘制聚类结果
    plt.scatter(data[:, 4], data[:, 5], c=cluster_assignments, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 4], kmeans.cluster_centers_[:, 5], s=200, c='red', marker='x',
                label='Centroids')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # 输出聚类结果
    # print(kmeans.labels_)

def dbscan():

    from sklearn.cluster import DBSCAN
    import numpy as np

    # 假设你有一个名为 data 的 10 个 64 维的张量
    # data = ...  # 填入你的张量
    data = load_np()[0]
    print(data.shape)
    # 将张量转换为NumPy数组
    # data_np = data.numpy()

    # 创建并拟合 DBSCAN 模型
    dbscan = DBSCAN(eps=0.2, min_samples=2)  # 这里的 eps 和 min_samples 是需要根据数据特点调节的参数
    clusters = dbscan.fit_predict(data)
    print(clusters)
    # clusters 中的每个元素即为对应数据点的簇标签


def func1():
    model = torch.load('./save_model/1.pt')
    test_loader = get_testloader()
    # train_loader = get_trainloader()
    # for (k, v), (k1, v1) in zip(model.state_dict().items(), model_h.state_dict().items()):
    #     if k[-6: ] == 'weight':
    #         print(torch.dist(v, v1, 2))
    test_loader = get_testloader()
    # train_loader = get_trainloader()
    valid_loss = 0
    round_acc = []
    label = [i for i in range(10)]
    for i in range(1, 2):
        # train_loader = get_data(i)
        # loss = nn.CrossEntropyLoss()
        # train(train_loader, model, loss, 1)

        confusion = ConfusionMatrix(num_classes=10, labels=label)
        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(test_loader):
                output = model(inputs)
                ret, predictions = torch.max(output.data, 1)
                confusion.update(predictions.cpu().numpy(), labels.cpu().numpy())

            confusion.plot()
            confusion.summary()


def func2(i):
    import torch
    data = get_means()
    max_val = np.amax(data)
    min_val = np.amin(data)
    # 给定的张量
    tensor = torch.tensor(data[i])

    # 找到最小值和最大值
    # min_val = tensor.min().item()
    # max_val = tensor.max().item()

    # 缩放到 0 到 256 的范围内
    scaled_tensor = ((tensor - min_val) / (max_val - min_val)) * 256

    # 转换为整数类型
    int_tensor = scaled_tensor.int()

    # 重新形状为表示图像的矩阵
    image_matrix = int_tensor.reshape(8, 8)

    print(image_matrix)

    # from PIL import Image
    #
    # import numpy as np
    #
    # im = Image.fromarray(np.uint8(image_matrix))
    # im.show("abc.png")

    import matplotlib.pyplot as plt

    # 假设 image_matrix 是你的图像矩阵
    plt.imshow(image_matrix, cmap='gray')
    plt.show()

def load_np():
    loaded_arrays = []
    with open('./det_k.npy', 'rb') as f:
        while True:
            try:
                loaded_array = np.load(f)
                loaded_arrays.append(loaded_array)
            except:
                break

                # 打印加载的数组
    # for array in loaded_arrays:
    # print(array.shape)
    print(len(loaded_arrays))
    return np.asarray(loaded_arrays)


from collections import defaultdict


def sample_dirichlet_train_data(n_client):
    alpha = 10000
    train_dataset = torchvision.datasets.CIFAR10(root='.data', train=True, download=False)
    # train_dataset = {i: i ** 2 for i in range(1000)}
    total_classes = dict()
    for ind, x in enumerate(train_dataset):
        _, label = x
        if label in total_classes:
            total_classes[label].append(ind)
        else:
            total_classes[label] = [ind]

    class_size = len(total_classes[0])
    per_client_list = defaultdict(list)
    n_class = len(total_classes.keys())

    np.random.seed(111)
    for n in range(n_class):
        random.shuffle(total_classes[n])
        n_party = n_client
        if True:
            sampled_probabilities = class_size * np.random.dirichlet(np.array(n_client * [alpha] + [alpha]))
            n_party = n_party + 1
            print(sampled_probabilities)
        else:
            sampled_probabilities = class_size * np.random.dirichlet(np.array(n_client * [alpha]))
        for p in range(n_party):
            n_image = int(round(sampled_probabilities[p]))
            sampled_list = total_classes[n][:min(len(total_classes[n]), n_image)]

            per_client_list[p].extend(sampled_list)
            # decrease the chosen samples
            total_classes[n] = total_classes[n][min(len(total_classes[n]), n_image):]

    # is a list to contain img_id
    return per_client_list

def ff():
    # 导入必要的库
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest

    # 创建一些示例数据
    rng = np.random.RandomState(42)
    # 生成一些正态分布的数据作为示例
    X = 0.3 * rng.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]

    # 创建模型并拟合数据
    clf = IsolationForest(random_state=rng)
    clf.fit(X_train)

    # 预测离群点
    y_pred = clf.predict(X_train)
    print(y_pred)

    # 可视化结果
    plt.title("IsolationForest")
    # 正常点为蓝色，离群点为红色
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == '__main__':
    k_means()