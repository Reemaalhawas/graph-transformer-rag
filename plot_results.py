import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    data_sets = [
        {"recall": [0.3336, 0.2332, 0.2158, 0.2048, 0.1568, 0.1584],
         "size": np.log([449, 8, 5, 5, 2, 2]), "colors": ["blue", "green", "green", "green", "red", "red"]},  # v0,3,6
        {"recall": [0.6219, 0.4919, 0.4433, 0.4049, 0.4047, 0.2734, 0.2716, 0.2691],
         "size": np.log([968, 214.51, 25, 20, 7, 2, 1.7, 1.8]),
         "colors": ["blue", "green", "green", "green", "green", "red", "red", "red"]},  # v4,8,9,10
        {"recall": [0.9262], "size": np.log([3746]), "colors": ["blue"]},
    ]

    # Plot each data point with specified color
    plt.figure(figsize=(8, 6))
    for i, data in enumerate(data_sets):
        x_pos = [i] * len(data["recall"])  # x-position for each set
        # Ensure the y (recall) values and sizes are the same length as x_pos
        for recall, size, color in zip(data["recall"], data["size"], data["colors"]):
            plt.scatter(
                [i], [recall], s=size*20, color=color, alpha=0.6  # Use lists for single points
            )

    # Labels and title
    plt.xlabel("Neo4j G-retriever pipelines")
    plt.ylabel("Recall")
    plt.xticks(range(len(data_sets)), [f"Set {i + 1}" for i in range(len(data_sets))])
    plt.title("Recall of various pipelines. Dot size = log(num_nodes)")

    # Create a legend for colors manually
    unique_colors = set()
    for data in data_sets:
        unique_colors.update(data["colors"])

    # Assign labels for the legend
    color_labels = {"blue": "Cypher retrieval", "green": "PCST output", "red": "G-retriever", "black": "Vector retrieval"}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=color_labels[color],
                          markerfacecolor=color) for color in unique_colors]

    plt.legend(handles=handles, title="Data Points", loc="upper left", bbox_to_anchor=(0, 1))
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # data = {
    #     "hit@1": [0.1, 0.5, 0.9],
    #     "recall": [0.2, 0.6, 0.8],
    #     "size": [50, 200, 300],
    #     "label": ["Cypher topk-1hop", "Point B", "Point C"],
    #     "color": ["blue", "green", "red"]
    # }
    #
    # # Plot
    # plt.figure(figsize=(8, 6))
    # for i in range(len(data["hit@1"])):
    #     plt.scatter(
    #         data["hit@1"][i], data["recall"][i],
    #         s=data["size"][i], color=data["color"][i], alpha=0.5, label=data["label"][i]
    #     )
    #
    # plt.xlabel("Hit@1")
    # plt.ylabel("Recall")
    # plt.title("Hit@1 vs Recall with Colored and Sized Points")
    #
    # # Add legend below the figure as a caption
    # plt.figlegend(
    #     loc="lower center", ncol=3, title="Data Points",
    #     bbox_to_anchor=(0.5, -0.15), frameon=False
    # )
    #
    # # Add custom caption below the legend
    # plt.figtext(0.5, -0.3,
    #             "Each point represents a unique entry with specified color and size based on data attributes.",
    #             ha="center", fontsize=10)
    #
    # plt.tight_layout()
    # plt.savefig("plot.png", dpi=300, bbox_inches="tight")
    # #plt.show()
