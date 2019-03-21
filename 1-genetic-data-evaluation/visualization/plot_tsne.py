import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def tsne_projection(X, y, perplexity):
    print('- Projecting by tSNE')
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=2000)
    X = tsne.fit_transform(X)

    Xf = X[y == 0]  # all projected samples with response = 0 (refractory)
    Xt = X[y == 1]  # all projected samples with response = 1 (responsive)

    print('- Plotting projected feats')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title('t-SNE: perplexity=%d' % perplexity)
    line1, = ax.plot(Xf[:,0], Xf[:,1], 'bo', linewidth=0.5, picker=5, label='Refractory')
    line2, = ax.plot(Xt[:,0], Xt[:,1], 'ro', linewidth=0.5, picker=5, label='Responsive')
    plt.legend(handles=[line1, line2])

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    # plt.scatter(X[:, 0], X[:, 1], c=y, marker='^')

    preds = np.random.choice([0, 1], size=(len(X),)) # mock predictions values

    fig, ax = plt.subplots()
    for i, (x, label, pred) in enumerate(zip(X, y, preds)):
        if label == 1 and label == pred:
            m = 'v'
            color = 'r'
        elif label == 1 and label != pred:
            m = 'v'
            color = 'b' 

        if label == 0 and label == pred:
            m = 'o'
            color = 'b'
        elif label == 0 and label != pred:
            m = 'o'
            color = 'r' 

        # m = 'v', color = 'r' if label == 1 and label == pred else  m = 'v', color = 'b'
        # m = 'o', color = 'b' if label == 0 and label == pred else  m = 'o', color = 'r'
        # ax.scatter(x[i,0], x[i,1], marker=marker)
        ax.scatter(x[0], x[1], marker=m, color=color)  

    plt.show()

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        sys.exit("python %s <preprocessed_dataset.csv> <(optional) perplexity (default=40)>" % sys.argv[0])

    csv_path = sys.argv[1]
    perplexity = 40
    if len(sys.argv) == 3:
        perplexity = int(sys.argv[2])

    print("- CSV: %s" % csv_path)
    print("- Perplexity: %d" % perplexity)
    print("----------------------\n")

    df = pd.read_csv(csv_path)
    X = df.values[:, 1:]
    y = df.values[:, 0]

    tsne_projection(X, y, perplexity)

if __name__ == "__main__":
    main()
