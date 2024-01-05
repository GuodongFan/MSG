import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from time import time
from matplotlib.font_manager import FontProperties


# 2.编写绘画函数，对输入的数据X进行画图。
def plot_embedding(X, y, title, ax):
    X = MinMaxScaler().fit_transform(X)

    for i in range(X.shape[0]):
        # plot every digit on the embedding
        ax.text(
            X[i, 0],
            X[i, 1],
            str(y[i]),
            #color=plt.cm.Set1(i),
            fontdict={"weight": "bold", "size": 10, 'horizontalalignment':'center', 'verticalalignment':'bottom'},
        )
        '''
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        ax.add_artist(imagebox)
        '''
    #ax.set_title(title)
    ax.axis("on")


# 3.选择要用那种方式对原始数据编码(Embedding),这里选择TSNE。
#  n_components = 2表示输出为2维，learning_rate默认是200.0,
embeddings = {
    "t-SNE embeedding": TSNE(
        n_components=2, init='pca', learning_rate=200.0, random_state=0, early_exaggeration=20, perplexity=3
    ),
}

def show_tsne2(X, y_map, show_api_list):
    # 4.根据字典里（这里只有TSNE）的编码方式，生成压缩后的编码矩阵
    # 即把每个样本生成了2维的表示。维度由原来的50位变成了2位。
    # Input: (n_sample, n_dimension)
    # Output: (n_sample, 2)
    show_api_set = set(show_api_list)

    X = X.detach().cpu().numpy()
    y = []
    y_idx = []
    for key, val in y_map.items():
        if key in show_api_set:
            y.append([key, val])
            y_idx.append(val)
    y.sort(key=lambda e:e[1], reverse=False)

    y = [key for key, val in y]
    X = X[[y_idx]]
    #X = X[0:10]
    #y = y[0:10]

    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        # 原作者的dict里有多种比较方法，我只用了t-SNE,需要的可以查询原链接：https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#manifold-learning-on-handwritten-digits-locally-linear-embedding-isomap
        if name.startswith("Linear Discriminant Analysis"):
            data = X.copy()
            data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        else:
            data = X

        print(f"Computing {name}...")
        start_time = time()
        print(data.shape, type(data.shape))
        projections[name] = transformer.fit_transform(data, y)
        timing[name] = time() - start_time



    # 6.把编码矩阵输出到二维图像中来。
    projections[name] = MinMaxScaler().fit_transform(projections[name])
    fig, ax = plt.subplots()
    title = f"{name} (time {timing[name]:.3f}s)"
    plot_embedding(projections[name], y, title, ax)

    color = list(range(0, 19))
    plt.scatter(projections[name][:,0], projections[name][:,1], c=color)

    #plt.ylabel('y', fontdict={'family': 'Times New Roman', 'size': 20})
    #plt.xlabel('x', fontdict={'family': 'Times New Roman', 'size': 20})
    fonten = FontProperties(fname='./times.ttf', size=18)
    plt.yticks(fontproperties=fonten)
    plt.xticks(fontproperties=fonten)
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }

    #plt.gcf().subplots_adjust(left=0.05, right=0.6, top=0.91, bottom=0.09)
    #plt.grid(True)
    plt.show()

def show_tsne(X, y_map):
    # 4.根据字典里（这里只有TSNE）的编码方式，生成压缩后的编码矩阵
    # 即把每个样本生成了2维的表示。维度由原来的50位变成了2位。
    # Input: (n_sample, n_dimension)
    # Output: (n_sample, 2)
    X = X.detach().cpu().numpy()
    y = []
    for key, val in y_map.items():
        y.append([key, val])
    y.sort(key=lambda e:e[1], reverse=False)

    y = [key for key, val in y]

    X = X[0:1000]
    y = y[0:1000]

    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        # 原作者的dict里有多种比较方法，我只用了t-SNE,需要的可以查询原链接：https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#manifold-learning-on-handwritten-digits-locally-linear-embedding-isomap
        if name.startswith("Linear Discriminant Analysis"):
            data = X.copy()
            data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
        else:
            data = X

        print(f"Computing {name}...")
        start_time = time()
        print(data.shape, type(data.shape))
        projections[name] = transformer.fit_transform(data, y)
        timing[name] = time() - start_time



    # 6.把编码矩阵输出到二维图像中来。
    projections[name] = MinMaxScaler().fit_transform(projections[name])
    fig, ax = plt.subplots()
    title = f"{name} (time {timing[name]:.3f}s)"
    plot_embedding(projections[name], y, title, ax)
    set(plt.gca, 'linewidth', 2, 'fontsize', 30, 'fontname', 'Times')
    error = data[:, 6]
    #plt.scatter(projections[name][:,0], projections[name][:,1], c=error)

    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }
    plt.tick_params(labelsize=10)
    #plt.gcf().subplots_adjust(left=0.05, right=0.6, top=0.91, bottom=0.09)
    plt.show()