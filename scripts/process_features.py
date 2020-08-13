import numpy as np
import lap
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import time
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
np.set_printoptions(precision=3)

def hex2rgb(h):
    # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    h = h.strip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# color_cycle = [hex2rgb(x) for x in color_cycle]
# print (color_cycle)


with open('/home/ed/Data/pickle/seq09_features.pkl', 'rb') as f:
    frames = pickle.load(f)

# Utility function to visualize the outputs of PCA and t-SNE
# https://www.datacamp.com/community/tutorials/introduction-t-sne
def visualize(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    # palette = color_cycle
    palette = np.array(sns.color_palette("hls", num_classes))
    print(np.squeeze([colors]))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[np.squeeze(colors)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()
    return f, ax, sc, txts

X = []
y = []
count = 0
for frame in frames:
    for id in frame:
        X.append(np.squeeze(frame[id]))
        y.append(id)
X = np.array(X)
y = np.array(y)
pca = PCA(n_components=10, svd_solver='auto')
pca_result = pca.fit_transform(X)
print (pca_result)
print(pca.explained_variance_ratio_)
TSNE(n_components=2).fit_transform(pca_result)
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
visualize(tsne_results, y)
# print (features)
quit()




print(frames)

quit()

for i in range(len(frames) - 1):
    print ('Comparison', i)
    curr = np.array([frames[i+1][x] for x in frames[i+1]])
    prev = np.array([frames[i][x] for x in frames[i]])
    curr = np.squeeze(curr)
    prev = np.squeeze(prev)
    dists = []
    for c in curr:
        row = []
        for p in prev:
            row.append(distance.cosine(c, p))
        dists.append(row)
    print (dists)
    cost, x, y = lap.lapjv(np.array(dists), extend_cost=True)
    print (cost)
    print (x)
    print (y)
    A = np.zeros_like(dists)
    for i in range(A.shape[0]):
        A[i, x[i]] = 1
    print (A)
    