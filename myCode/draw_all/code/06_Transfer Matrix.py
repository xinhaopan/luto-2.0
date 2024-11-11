import matplotlib.patches as mpatches
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import pandas as pd

from tools.parameters import *
from tools.data_helper import *



def calculate_transition_matrix(land_use_data_path, mapping_data_path, area_column='Area (ha)',
                                        from_column='From land-use', to_column='To land-use',
                                        desc_column='desc', ag_non_ag_group_column='ag_non_ag_group'):
    """
    Calculate a transition matrix for land-use changes, ordered by the 'ag_non_ag_group' column in the mapping data.
    NaN values are removed from the final transition matrix.

    Parameters:
    - land_use_data_path (str): Path to the CSV file containing land-use data with 'From land-use', 'To land-use', and 'Area (ha)' columns.
    - mapping_data_path (str): Path to the CSV file containing the mapping with 'desc' and 'ag_non_ag_group' columns.
    - area_column (str): Column name for area values. Default is 'Area (ha)'.
    - from_column (str): Column name for the initial land-use type. Default is 'From land-use'.
    - to_column (str): Column name for the target land-use type. Default is 'To land-use'.
    - desc_column (str): Column name in the mapping file for detailed land-use types. Default is 'desc'.
    - ag_non_ag_group_column (str): Column name in the mapping file for grouped land-use types. Default is 'ag_non_ag_group'.

    Returns:
    - pd.DataFrame: Transition matrix with aggregated land-use change areas, ordered by the specified grouping column.
    """
    # Load the main data and the mapping table
    land_use_data = pd.read_csv(land_use_data_path)
    mapping_data = pd.read_excel(mapping_data_path)

    # Create dictionaries for grouping based on mapping
    group_mapping = mapping_data.set_index(desc_column)[ag_non_ag_group_column].to_dict()
    ordered_groups = mapping_data[ag_non_ag_group_column].dropna().drop_duplicates().tolist()

    # Map the grouping to 'From' and 'To' columns
    land_use_data['From_group'] = land_use_data[from_column].map(group_mapping)
    land_use_data['To_group'] = land_use_data[to_column].map(group_mapping)

    # Drop rows where mapping was not found
    land_use_data.dropna(subset=['From_group', 'To_group'], inplace=True)

    # Group by the new grouped land-use types and sum the areas
    transition_matrix = land_use_data.groupby(['From_group', 'To_group'])[area_column].sum().unstack(fill_value=0)

    # Reindex the matrix to ensure it follows the order in `ag_non_ag_group`
    transition_matrix = transition_matrix.reindex(index=ordered_groups, columns=ordered_groups, fill_value=0)

    return transition_matrix

def plot_all_transition_matrices(matrices, labels_list, label_mapping, output_path="transition_matrix_all.png", font_size=15):
    """
    Plots multiple transition matrices with shorthand labels on a grid layout.

    Parameters:
    - matrices: List[np.array or pd.DataFrame], transition matrices to plot.
    - labels_list: List[List[str]], list of label sets (one for each matrix).
    - label_mapping: Dict[str, str], maps full land-use names to shorthand labels.
    - output_path: str, path to save the output image.
    - font_size: int, font size for labels and tick marks.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.subplots_adjust(hspace=0.15, wspace=0.15)

    for i, (matrix, labels) in enumerate(zip(matrices, cycle(labels_list))):
        # Ensure matrix is a NumPy array for imshow
        if isinstance(matrix, pd.DataFrame):
            full_labels = matrix.columns.tolist()
            short_labels = [label_mapping.get(label, label) for label in full_labels]
            matrix = matrix.values
        else:
            short_labels = [label_mapping.get(label, label) for label in labels]

        ax = axes[i // 3, i % 3]
        cax = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')

        # Set the tick labels
        ax.set_xticks(np.arange(len(short_labels)))
        ax.set_yticks(np.arange(len(short_labels)))
        ax.set_xticklabels(short_labels, ha="center", fontsize=font_size)
        ax.set_yticklabels(short_labels, fontsize=font_size)

        # Display values within each cell
        for x in range(len(short_labels)):
            for y in range(len(short_labels)):
                value = matrix[x, y]
                color = "white" if value > np.max(matrix) / 2 else "black"
                ax.text(y, x, f"{value:.1f}", ha="center", va="center", color=color, fontsize=font_size)

    # Add color bar
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.02])
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Transition Area (M ha)', fontsize=font_size+5)
    cbar.ax.tick_params(labelsize=font_size+5)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

mapping_data_path = 'tools/land use group.xlsx'
matrices = []
for input_dir in input_files:
    file_dir = get_path(input_dir)
    land_use_data_path = file_dir + '/out_2050/transition_matrix_2010_2050.csv'
    matrice = calculate_transition_matrix(land_use_data_path, mapping_data_path)
    matrices.append(matrice / 1e6)
labels = list(matrice)

plt.rcParams['font.family'] = 'Arial'
label_mapping = {
    'Crops': 'C',
    'Livestock': 'L',
    'Unallocated - modified land': 'UM',
    'Unallocated - natural land': 'UN',
    'Non-agricultural land-use': 'NA'
}
plot_all_transition_matrices(matrices, labels,label_mapping, output_path="../output/06_transition_matrix_all.png")
