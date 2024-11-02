import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


if __name__ == "__main__":

    data = pd.DataFrame({
        'precision': [1.85, 5.1, 30.1, 4.79, 6.54, 3.53, 18.27, 4.36, 0.27, 3.8],
        'recall': [62.2, 44.3, 27.3, 33.3, 23.9, 49.1, 20.24, 43.49, 92.6, 66],
        'num_nodes': [968, 25, 1.8, 449, 7, 214, 2, 20.7, 3746, 720],
        #'pipelines': [['A', 'C'], ['A'], ['A'], ['B'], ['C'], ['D'], ['D'], ['E'], ['F'], 'G'],
        'color': ['blue', 'green', 'red', 'blue', 'green', 'green', 'red', 'green', 'blue', 'blue']
        # v8cypher, v8pcst, v8gretriever v0cypher, v4pcst, v9pcst, v9gretriever, v10pcst, llmcypherm, llmcypher2
    })

    plt.figure(figsize=(8, 6))

    # Plot each point with specified colors and annotate with `num_nodes`
    for idx, row in data.iterrows():
        plt.scatter(row['precision'], row['recall'], s=20*np.log(row['num_nodes']), color=row['color'], alpha=0.6)
        plt.text(row['precision'], row['recall'], str(row['num_nodes']), fontsize=9, ha='right')

    # Plot smooth connecting lines for each unique pipeline
    # unique_pipelines = set(pipeline for sublist in data['pipelines'] for pipeline in sublist)
    # for pipeline in unique_pipelines:
    #     pipeline_data = data[data['pipelines'].apply(lambda x: pipeline in x)]
    #     if len(pipeline_data) > 1:  # Ensure there are enough points to create a curve
    #         # Sort data by precision for smoother plotting
    #         pipeline_data = pipeline_data.sort_values('precision')
    #
    #         # Smooth curve using spline interpolation
    #         # x_smooth = np.linspace(pipeline_data['precision'].min(), pipeline_data['precision'].max(), 300)
    #         # spline = make_interp_spline(pipeline_data['precision'], pipeline_data['recall'], k=2)
    #         # y_smooth = spline(x_smooth)
    #         # plt.plot(x_smooth, y_smooth, linestyle='-', linewidth=1, label=f'Pipeline {pipeline}')
    #
    #         plt.plot(pipeline_data['precision'], pipeline_data['recall'], linestyle='-', linewidth=1,
    #                  label=f'Pipeline {pipeline}')

    # Labels, legend, and title
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall plots with num_nodes")
    color_labels = {"blue": "Cypher retrieval", "green": "Subgraph pruning", "red": "Tuned GNN+LLM"}
    unique_colors = ["blue", "green", "red"]

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=color_labels[color],
                          markerfacecolor=color) for color in unique_colors]

    plt.legend(handles=handles, loc="upper right")

    plt.savefig("plotpr.png", dpi=300, bbox_inches="tight")
