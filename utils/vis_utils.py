import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap.plots
from sklearn.manifold import TSNE

LEGAL_STATUS_COLORS = {
    "markers": {
        "DEC": "orange",
        "ILL": "darkred",
        "MED": "navy",
        "LGL": "darkgreen"
    },
    "errors": {
        "DEC": "wheat",
        "ILL": "red",
        "MED": "blue",
        "LGL": "limegreen"
    }
}

PADDING = 7e-4

COUNTRIES_TO_SHOW = {'Estonia', 'Mexico', 'Uruguay', 'Hungary', 'Gibraltar', 'Belarus', 'Montenegro', 'Venezuela',
                     'Indonesia', 'Latvia', 'Saudi Arabia', 'Norway', 'Portugal', 'Ecuador', 'Lebanon',
                     'Czech Republic', 'Cyprus', 'United Kingdom', 'Belgium', 'Turkey', 'Slovakia', 'Poland',
                     'Austria', 'Italy', 'Croatia', 'Netherlands'}

def plot_regular_barplot(x: Union[list, np.ndarray], y: Union[list, np.ndarray], save_path: Union[str, None] = None,
                         **kwargs):
    """
    Creates a regular bar plot using the provided x and y data.

    Parameters
    ----------
    x : list or np.ndarray
        The categories or x-axis values for the bar plot.
    y : list or np.ndarray
        The heights or y-axis values for the bars.
    save_path : str or None, optional
        File path to save the plot image. If None (default), the plot is not saved.
    **kwargs
        Additional keyword arguments passed to the plotting function, such as:
            - figsize: tuple, figure size
            - color: str, bar color
            - xlabel, ylabel, title: str, axis and plot labels
            - label_fontdict: dict, font properties for axis labels
            - title_fontsize: int, font size for the title
            - xticks, yticks: list, custom tick locations
            - xticks_fontsize, yticks_fontsize: int, font size for ticks
            - xticks_rotation, yticks_rotation: int or float, rotation for ticks
            - legend: bool, whether to show legend
            - legend_loc: str, legend location
            - legend_fontsize: int, legend font size
            - is_tight_layout: bool, whether to use tight layout
    """
    plt.figure(figsize=kwargs.get('figsize', (10, 10)))
    plt.bar(x, y, color=kwargs.get('color', 'blue'))
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else None,
               fontdict=kwargs.get('label_fontdict', None))
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else None,
               fontdict=kwargs.get('label_fontdict', None))
    plt.title(kwargs['title'] if 'title' in kwargs else None,
              fontsize=kwargs['title_fontsize'] if 'fontsize' in kwargs else None)
    plt.xticks(kwargs['xticks'] if 'xticks' in kwargs else None,
               fontsize=kwargs['xticks_fontsize'] if 'xticks_fontsize' in kwargs else None,
               rotation=kwargs['xticks_rotation'] if 'xticks_rotation' in kwargs else None)
    plt.yticks(kwargs['yticks'] if 'yticks' in kwargs else None,
               fontsize=kwargs['yticks_fontsize'] if 'yticks_fontsize' in kwargs else None,
               rotation=kwargs['yticks_rotation'] if 'yticks_rotation' in kwargs else None)
    if kwargs.get('legend', False):
        plt.legend(loc=kwargs['legend_loc'] if 'legend_loc' in kwargs else None,
                   fontsize=kwargs['legend_fontsize'] if 'legend_fontsize' in kwargs else None)
    if kwargs.get('is_tight_layout', False):
        plt.tight_layout()
    if save_path is not None:
        if os.path.exists(save_path):
            raise FileExistsError(save_path)
        plt.savefig(save_path)
    plt.show()


def plot_model_predictions(
        gt_values: Union[np.ndarray, pd.Series],
        lasso_predictions: Union[np.ndarray, pd.Series, None],
        ridge_predictions: Union[np.ndarray, pd.Series, None],
        rf_predictions: Union[np.ndarray, pd.Series, None],
        x_tick_labels: list[str] = None,
        title: str = "Model Predictions vs. Ground Truth",
        save_path: Union[str, None] = None
):
    """Plots model predictions against Ground Truth (GT) values on a line plot.

    The GT values are emphasized with a larger marker size to stand out. Each
    model's predictions are plotted with a unique color, marker, and line style.

    Args:
        gt_values (Union[np.ndarray, pd.Series]): The ground truth values.
        lasso_predictions (Union[np.ndarray, pd.Series, None]): Predictions from the Lasso model.
        ridge_predictions (Union[np.ndarray, pd.Series, None]): Predictions from the Ridge model.
        rf_predictions (Union[np.ndarray, pd.Series, None]): Predictions from the Random Forest model.
        x_tick_labels (list[str], optional): Custom labels for the x-axis ticks. Defaults to None.
        title (str, optional): The title for the plot. Defaults to "Model Predictions vs. Ground Truth".
        save_path (Union[str, None], optional): File path to save the plot. If None, the plot is not saved.
                                                Defaults to None.
    """
    x_indices = np.arange(len(gt_values))  # X-axis values, e.g., data point indices

    plt.figure(figsize=(12, 7))  # Set the figure size for better readability

    # Plot Ground Truth values - larger marker size to stand out
    plt.plot(
        x_indices,
        gt_values,
        label='Ground Truth',
        color='black',  # Distinct color
        marker='o',  # Circle marker
        linestyle='-',  # Solid line
        linewidth=2,
        markersize=10  # Larger marker size
    )

    # Plot Lasso Regression predictions
    if lasso_predictions is not None:
        plt.plot(
            x_indices,
            lasso_predictions,
            label='Lasso Predictions',
            color='blue',  # Different color
            marker='x',  # 'x' marker
            linestyle='--',  # Dashed line
            markersize=7  # Default marker size
        )

    # Plot Ridge Regression predictions
    if ridge_predictions is not None:
        plt.plot(
            x_indices,
            ridge_predictions,
            label='Ridge Predictions',
            color='green',  # Different color
            marker='s',  # Square marker
            linestyle=':',  # Dotted line
            markersize=7  # Default marker size
        )

    # Plot Random Forest predictions
    if rf_predictions is not None:
        plt.plot(
            x_indices,
            rf_predictions,
            label='Random Forest Predictions',
            color='red',  # Different color
            marker='^',  # Triangle up marker
            linestyle='-.',  # Dash-dot line
            markersize=7  # Default marker size
        )

    # Set custom x-ticks if provided and valid
    if x_tick_labels is not None:
        plt.xticks(x_indices, x_tick_labels, rotation=45, ha='right')  # Rotate labels for better visibility

    plt.title(title)  # Title of the plot
    plt.xlabel('Country (Cannabis legal status)')  # X-axis label
    plt.ylabel('total treatment cases')  # Y-axis label
    plt.legend()  # Display the legend
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for readability
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    if save_path is not None:
        plt.savefig(save_path)  # Save the plot if a path is provided
    plt.show()  # Display the plot


def plot_legal_cat_pred_errors(x_data: pd.DataFrame, y_trues: Union[pd.Series, pd.DataFrame, np.ndarray],
                               y_preds: Union[pd.Series, pd.DataFrame, np.ndarray], predictor_name: str,
                               drug_group_name: str, save_path: Union[str, None] = None):
    """
    Plots prediction errors for different legal categories.

    Args:
        x_data (pd.DataFrame): DataFrame containing features, including 'legal_category'.
        y_trues (Union[pd.Series, pd.DataFrame, np.ndarray]): True target values.
        y_preds (Union[pd.Series, pd.DataFrame, np.ndarray]): Predicted values.
        predictor_name (str): Name of the predictor (for plot title).
        drug_group_name (str): Name of the drug group (for plot title).
        save_path (str): File path to save the plot image. If None (default), the plot is not saved.
    """
    data_for_errbar = x_data.copy()
    data_for_errbar["y_true"] = y_trues
    data_for_errbar["y_pred"] = y_preds
    data_for_error_plot = {}

    for legal_status, group in data_for_errbar.groupby('legal_category'):
        group = group.sort_values('y_true')
        y_pred_below = []
        y_pred_above = []
        for _, row in group.iterrows():
            if row['y_pred'] > row['y_true']:
                y_pred_above.append(row['y_pred'] - row['y_true'])
                y_pred_below.append(0)
            else:
                y_pred_below.append(row['y_true'] - row['y_pred'])
                y_pred_above.append(0)
        data_for_error_plot[legal_status] = {
            "y_true": group['y_true'].values.tolist(),
            "y_pred_above": y_pred_above,
            "y_pred_below": y_pred_below
        }

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    marker_color = ['orange', 'darkred', 'navy', 'darkgreen']
    error_colors = ['wheat', 'red', 'blue', 'limegreen']

    for i, legal_status in enumerate(data_for_error_plot):
        y_true = data_for_error_plot[legal_status]['y_true']
        above_y, below_y = data_for_error_plot[legal_status]['y_pred_above'], data_for_error_plot[legal_status][
            'y_pred_below']

        # Determine the correct subplot for the current legal_status
        row_idx = 0 if i < 2 else 1
        col_idx = i % 2

        ax[row_idx][col_idx].plot(range(len(y_true)), y_true, marker='o', color=marker_color[i], label=legal_status)
        ax[row_idx][col_idx].errorbar(range(len(y_true)), y_true, [below_y, above_y], alpha=0.8, ecolor=error_colors[i])
        ax[row_idx][col_idx].set_title(f"Errors for legal status {legal_status}")
        ax[row_idx][col_idx].set_yscale('log')  # Apply log scale to each subplot

    plt.suptitle(f"{predictor_name} {drug_group_name} treatment per capita prediction values per Legalization Category",
                 fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.98])
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_error_scatter(y_true: Union[pd.Series, np.ndarray],
                       y_preds: Union[pd.Series, np.ndarray],
                       legal_statuses: Union[pd.Series, list[str], np.ndarray],
                       countries: Union[pd.Series, list[str], np.ndarray],
                       title: str = None,
                       save_path: str = None):
    """
    Plots a scatter plot of true vs. predicted values with error bars,
    averaging the results for each country.

    Args:
        y_true (Union[pd.Series, np.ndarray]): Ground truth values.
        y_preds (Union[pd.Series, np.ndarray]): Predicted values. Expected to be an iterable of iterables (e.g., list of lists).
        legal_statuses (Union[pd.Series, list[str], np.ndarray]): Legal status for each data point.
        countries (Union[pd.Series, list[str], np.ndarray]): Country for each data point.
        title (str, optional): The title for the plot. Defaults to None.
        save_path (str, optional): The file path to save the plot. Defaults to None.
    """

    df = pd.DataFrame({
        'y_true': y_true,
        'pred_mean': [np.mean(p) for p in y_preds],
        'pred_min': [np.min(p) for p in y_preds],
        'pred_max': [np.max(p) for p in y_preds],
        'legal_status': legal_statuses,
        'country': countries
    })

    agg_df = df.groupby('country').agg(
        y_true_mean=('y_true', 'mean'),
        pred_mean_avg=('pred_mean', 'mean'),
        pred_min_overall=('pred_min', 'min'),
        pred_max_overall=('pred_max', 'max'),
        legal_status=('legal_status', 'first')  # Take the first legal status for the country
    ).reset_index()

    for _, row in agg_df.iterrows():
        country_name = row['country']
        if country_name not in COUNTRIES_TO_SHOW:
            continue
        legal_status = row['legal_status']
        y_t = row['y_true_mean']
        y_p_mean = row['pred_mean_avg']

        # Calculate error bar lengths from the aggregated min/max
        y_err_lower = y_p_mean - row['pred_min_overall']
        y_err_upper = row['pred_max_overall'] - y_p_mean
        y_err = np.array([[y_err_lower], [y_err_upper]])

        plt.errorbar(x=y_t, y=y_p_mean, yerr=y_err,
                     ecolor=LEGAL_STATUS_COLORS["errors"].get(legal_status, 'gray'),
                     marker='o',
                     color=LEGAL_STATUS_COLORS["markers"].get(legal_status, 'black'),
                     label=f"{country_name}--{legal_status}",
                     linestyle='None')

    # plot ideal x=y line - prediction = true value
    plt.plot([0, agg_df['y_true_mean'].mean() + PADDING], [0, agg_df['pred_mean_avg'].mean() + PADDING], 'r--')

    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), prop={'size': 8})
    plt.xlabel("Treatments per capita - Ground Truth (Averaged by Country)")
    plt.ylabel("Treatments per capita - Prediction Mean (Averaged by Country)")
    plt.title(title if title is not None else "Overall averaged Predictions vs. Ground Truth by Country")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, title: str, colors: str = 'coolwarm', save_path: str = None):
    """Plots a correlation matrix heatmap for numerical features in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title for the heatmap plot.
        colors (str, optional): The colormap for the heatmap. Defaults to 'coolwarm'.
        save_path (str, optional): The file path to save the plot. Defaults to None.
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['number'])

    if numerical_df.empty:
        print("No numerical features found in the DataFrame to plot a correlation matrix.")
        return

    # Calculate the correlation matrix
    correlation_matrix = numerical_df.corr()

    # Plot the correlation matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap=colors, fmt=".2f", linewidths=.5)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_tsne(df_scaled: pd.DataFrame, color_data: Union[pd.Series, np.ndarray], title: str = '', perplexity: int = 30,
              random_state: int = 42, legend_title: str = None, colorbar: bool = False) -> None:
    """
    Plots t-SNE visualization of the given scaled data.

    Args:
        df_scaled (pd.DataFrame): The preprocessed and scaled DataFrame ready for t-SNE.
        color_data (pd.Series): A Series containing the data to be used for coloring points (e.g., 'total_treatments' or
                                 'legal_category'). Its index should align with df_scaled's index.
        title (str): Title of the plot.
        perplexity (int): Perplexity parameter for t-SNE.
        random_state (int): Random state for reproducibility.
        legend_title (str): Title for the legend or colorbar.
        colorbar (bool): If True, attempts to display a colorbar (for numerical color_data).
                         If False, displays a standard legend (for categorical or if colorbar not desired).
    """
    # Adjust perplexity based on data size to avoid error
    actual_perplexity = min(perplexity, len(df_scaled) - 1)
    if actual_perplexity <= 0:
        print("Not enough data points for t-SNE with a positive perplexity. Skipping plot.")
        return

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=actual_perplexity)
    tsne_results = tsne.fit_transform(df_scaled)

    # Create a DataFrame for t-SNE results
    tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne_component_1', 'tsne_component_2'], index=df_scaled.index)
    if isinstance(color_data, pd.Series):
        tsne_df['coloring'] = color_data.reindex(tsne_df.index) # Ensure color_data aligns with tsne_df index
    else:
        tsne_df['coloring'] = color_data.reshape((-1, 1))

    # Plotting
    plt.figure(figsize=(16, 14)) # Increased figure size
    is_numerical_hue = pd.api.types.is_numeric_dtype(tsne_df['coloring'])

    ax = sns.scatterplot( # Assign the Axes object to 'ax'
        x='tsne_component_1',
        y='tsne_component_2',
        hue='coloring',
        palette='viridis' if is_numerical_hue else 'tab10', # Use viridis for numerical, tab10 for categorical
        data=tsne_df,
        legend='full', # This will make seaborn automatically create a colorbar/legend
        s=200, # Increased point size
        alpha=0.7
    )

    plt.title(title, fontsize=22) # Increased title font size
    plt.xlabel('t-SNE Component 1', fontsize=20) # Increased x-axis label font size
    plt.ylabel('t-SNE Component 2', fontsize=20) # Increased y-axis label font size
    plt.xticks(fontsize=18) # Increased x-axis tick label font size
    plt.yticks(fontsize=18) # Increased y-axis tick label font size

    if colorbar and is_numerical_hue:
        # Explicitly create and configure the colorbar using plt.colorbar
        # This is more robust than relying on ax.collections[0].colorbar
        norm = plt.Normalize(tsne_df['coloring'].min(), tsne_df['coloring'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, orientation='vertical') # Use 'ax' to associate with the subplot
        cbar.set_label(legend_title if legend_title else 'Value', fontsize=20)
        cbar.ax.tick_params(labelsize=18)
        ax.get_legend().remove()
    else: # For categorical hue, or if colorbar=False
        if ax.legend_ is not None:
            legend = ax.legend_
            plt.setp(legend.get_title(), fontsize=20)
            plt.setp(legend.get_texts(), fontsize=18)
            if legend_title:
                legend.set_title(legend_title) # Set legend title

    plt.grid(True)
    plt.tight_layout()
    plt.show() # Display the plot


def plot_rf_feature_importance(feature_importance: Union[pd.Series, np.ndarray, list[np.float64]],
                               errs: Union[pd.Series, np.ndarray, list[np.float64]],
                               feature_names: Union[pd.Series, np.ndarray, list[str]],
                               title: str, save_path: str):
    """Plots feature importances from a Random Forest model with error bars.

    This function creates a bar plot to visualize the mean decrease in impurity (MDI)
    for each feature, with error bars representing the standard deviation across trees.
    (Based on scikit-learn's example:
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

    Args:
        feature_importance (Union[pd.Series, np.ndarray, list[np.float64]]):
            The feature importance values (e.g., `model.feature_importances_`).
        errs (Union[pd.Series, np.ndarray, list[np.float64]]):
            The standard deviation of the importances across the trees in the forest.
        feature_names (Union[pd.Series, np.ndarray, list[str]]):
            The names of the features corresponding to the importance values.
        title (str): The super title for the entire figure.
        save_path (str): The file path to save the plot.
    """

    forest_importances = pd.Series(feature_importance, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=errs, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_shapley_feature_importance(shap_values: shap.Explanation, title: str, save_path: Union[str, None] = None):
    """Plots a SHAP waterfall plot to explain a single model prediction.

        This function visualizes how each feature contributes to pushing the model's
        output from the base value to the final prediction for a specific instance.
        It specifically plots the explanation for the first instance (`shap_values[0]`).

        Args:
            shap_values (shap.Explanation): A SHAP Explanation object, typically the
                                            output of `shap.Explainer(model)(X)`.
            title (str): The title for the plot.
            save_path (Union[str, None], optional): The file path to save the plot.
    """
    ax = shap.plots.waterfall(shap_values[0], show=False)
    ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()



