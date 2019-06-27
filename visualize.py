from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from data import *

def tSNE_raw():
    tr_x, tr_y, tst_x, tst_y = dataloader()
    # TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
    #      learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
    #      min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
    #      random_state=None, method='barnes_hut', angle=0.5)
    tsne = TSNE(n_components=2)
    y = tsne.fit_transform(tst_x)
    np.save(y, './save/mnist_test_raw_tSNE')
    print(y)

def plot():
    y = np.load('./save/mnist_test_raw_tSNE')
    plt.plot(y)
    # plt.xlabel('City')
    # plt.ylabel('Response')
    # plt.title('Experiment Result')
    plt.show()

if __name__ == "__main__":
    tSNE_raw()
    # plot()
    pass