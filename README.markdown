# Energy Data Analysis and Modeling

## Project Overview

This project analyzes energy consumption data through exploratory data analysis (EDA), preprocessing, feature engineering, and machine learning modeling. It uses Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, and Streamlit to process data, train models, and deploy an interactive web app.

## Setup Instructions

### Prerequisites

- Python
- pip
- Jupyter Notebook
- Virtual environment (recommended)

### Installation

1. **Clone Repository** (if applicable):

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. **Create Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Code

1. **Launch Jupyter Notebook**:

   ```bash
   cd <project-folder>
   jupyter notebook
   ```
2. **EDA and Preprocessing**:
   - Open `EDA&Data_Preprocessing.ipynb`.
   - Run all cells.
   - Outputs `preprocessed_energy_data.csv` in `data/preprocessed/`.
3. **Feature Engineering**:
   - Open `Feature_Engineering.ipynb`.
   - Run all cells.
   - Outputs `combined_data.csv` in `data/processed/`.
4. **Modeling**:
   - Open `Modeling.ipynb`.
   - Run all cells.
   - Saves models in `models/`.
5. **Run Streamlit App**:

   ```bash
   streamlit run app.py
   ```

## Notes

- Ensure libraries are installed which mentioned in requirements.txt file, to avoid errors.
- Verify file paths in notebooks if data directories differ.