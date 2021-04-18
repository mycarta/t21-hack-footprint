import matplotlib.pyplot as plt
from skimage.util import img_as_float


def view_difference(ori, new):
    # display results
    fig, axes = plt.subplots(1, 3, figsize=(10, 20), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(ori, cmap="gray", aspect=0.5, origin="lower", interpolation="none")
    ax[0].axis("off")
    ax[0].set_title("2D", fontsize=24)

    ax[1].imshow(new, cmap="gray", aspect=0.5, origin="lower", interpolation="none")
    ax[1].axis("off")
    ax[1].set_title("3D", fontsize=24)

    diff = img_as_float(ori - new)

    ax[2].imshow(diff, cmap="gray", aspect=0.5, origin="lower", interpolation="none")
    ax[2].axis("off")
    ax[2].set_title("Difference", fontsize=24)

    fig.tight_layout()
    plt.show()
