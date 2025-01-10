
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def draw_concurrency_matrix(label, sets, categories=None, figsize=(8, 8), cmap='viridis'):
    """
    Draws a concurrency matrix for a list of sets.

    Parameters:
        sets (list of set): A list of sets, where each set contains categories.
        categories (list or None): An optional list of categories to include in the matrix.
                                  If None, all unique categories in the sets are used.
        figsize (tuple): Figure size for the plot.
        cmap (str): Colormap to use for the matrix.
    """
    # Determine the unique categories
    if categories is None:
        categories = sorted(set().union(*sets))
    
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    n_categories = len(categories)

    # Initialize the concurrency matrix
    matrix = np.zeros((n_categories, n_categories), dtype=int)

    # Fill the matrix with concurrency counts
    for item_set in sets:
        indices = [category_to_idx[category] for category in item_set if category in category_to_idx]
        for i in indices:
            for j in indices:
                matrix[i, j] += 1

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Co-occurrence Count", rotation=-90, va="bottom")

    # Label the axes
    ax.set_xticks(np.arange(n_categories))
    ax.set_yticks(np.arange(n_categories))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels(categories)

    # Add gridlines and adjust layout
    ax.set_xticks(np.arange(-0.5, n_categories, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_categories, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Concurrency Matrix")
    plt.tight_layout()
    plt.savefig(label)


if __name__ == '__main__':

    # Example usage
    sets = [
        {"A", "B", "C"},
        {"A", "C"},
        {"B", "C", "D"},
        {"A", "D"},
    ]
    
    categories = ["A", "B", "C", "D"]  # Optional, specify categories to include in the matrix
    draw_concurrency_matrix('test.png', sets, categories)
