# ds-lab-2025


# Cannabis legalization effects on hard drugs usage and treatments

#### background:
According to the 2023 World Drug Report, global drug use has surged by 23% in the last decade, now affecting roughly 296 million people.
This rise places immense strain on healthcare systems and challenges law enforcement and policymakers worldwide.

In response, nations have adopted a wide spectrum of policies, from strict enforcement to decriminalization and harm reduction strategies.
A particularly notable policy shift has been the move towards cannabis legalization in various forms.
However, the downstream effects of these policies, especially on the use of other illicit substances, are not yet fully understood.

This project aims to contribute to this evidence base by analyzing public data from various countries.
By comparing different national contexts and policy landscapes, we seek to uncover potential correlations and provide data-driven insights for policymakers.


#### Objectives:
- Build statistical models around available public data and try to estimate the number of treatments required for different countries for a hard drug in a year.
- Look whether legislation related to Cannabis legalization might affect usage of different drugs in those countries.

## Repository Structure

-   **`data/`**: Contains all project data, separated into 'raw' for original, untouched datasets and `final_dataframe.csv` for cleaned, transformed, and merged datasets ready for modeling.
-   **`individual_dataset_notebooks/`**: Holds Jupyter notebooks used for the initial exploration and preprocessing of each raw dataset individually before they are combined in the main preprocessing notebook.
-   **`utils/`**: A Python package containing helper modules.
    -   `vis_utils.py`: A utility module with reusable functions for creating various plots and visualizations, helping to keep the main notebooks cleaner and more focused on analysis.
    -   `models.py`: A module containing definitions to classes of models, with functionalities like fit, predict, etc., moreover, this module also contains the evaluation calculation methods.
    -   `dataset_processing_utils.py`: A module containing helper functions for building and manipulating dataframes, and used across all notebooks.
-   **`data_preprocessing.ipynb`**: The primary Jupyter notebook that orchestrates the merging and final preparation of all individual datasets into a single, model-ready dataset.
-   **`modelling.ipynb`**: The main Jupyter notebook for building, training, evaluating, and interpreting machine learning models.
-   **`requirements.txt`**: A file listing all Python packages and their versions required to create the correct environment and run the project code.
-   **`README.md`**: This file, providing an overview of the project, setup instructions, and repository structure.

---

## Instructions:
1. Change directory to where you cloned this repo - `cd /wherever/this/repo/is/saved/on/your/computer/`
2. Install a 3.10 Python environment with a package manager of your choice (`venv`, `virtualenv`, `conda`, etc.). **Note** - use the `requirements.txt` file to reference needed packages. 
3. Run your Jupyter server - `jupyter notebook`
4. Open and run the `data_preprocessing.ipynb` notebook. (You can run individual dataset processing notebooks under the `individual_dataset_notebooks` folder)
5. Afterwards, run the `modelling.ipynb` notebook for models and evaluation.
