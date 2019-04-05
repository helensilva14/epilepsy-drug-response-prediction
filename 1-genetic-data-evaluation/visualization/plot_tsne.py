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
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    ax1.set_title('t-SNE using the original sample labels (perplexity=%d)' % perplexity)
    line1, = ax1.plot(Xf[:,0], Xf[:,1], 'bo', linewidth=0.5, picker=5, label='Refractory')
    line2, = ax1.plot(Xt[:,0], Xt[:,1], 'ro', linewidth=0.5, picker=5, label='Responsive')
    plt.legend(handles=[line1, line2])

    preds = np.random.choice([0, 1], size=(len(X),)) # mock predictions values

    # fig2 = plt.figure(figsize=(10, 10))
    # ax2 = fig2.add_subplot(121)

    fig, ax2 = plt.subplots()
    ax2.set_title('t-SNE plotting the original vs predicted sample labels (perplexity=%d)' % perplexity)
    for i, (x, label, pred) in enumerate(zip(X, y, preds)):
        color = 'r' if label == 1 and label == pred else 'b' 
        color = 'b' if label == 0 and label == pred else 'r' 

        ax2.scatter(x[0], x[1], marker='v' if label == 1 else 'o', color=color)  

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
