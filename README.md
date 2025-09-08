# Predict Podcast Listening Time

## Project Overview
As a podcast enthusiast who listens to several episodes every week, I was inspired to build a machine learning model to predict podcast listening time. This project was developed as part of a Kaggle competition, where the goal is to forecast how long users might listen to a podcast episode based on key features. These features include:

- **Episode_Length_minutes**: The total duration of the episode in minutes.
- **Genre**: The category of the podcast (e.g., news, comedy, education).
- **Host_Popularity_percentage**: A score representing the host's popularity (0-100%).
- **Publication_Day**: The day of the week the episode was published.
- **Publication_Time**: The time of day the episode was released.
- **Guest_Popularity_percentage**: A score for any guest's popularity (0-100%).

The model achieved strong performance with a Mean Absolute Error (MAE) of 12.96, a low Mean Squared Error (MSE), and an R² score close to 1, indicating it accurately captures the relationships in the data. This project demonstrates a practical application of regression modeling for time prediction, and it's a great learning exercise for beginners in machine learning.

You can explore the full code on my GitHub repository [link to repo]. Below, I'll walk you through the step-by-step process, including key decisions, challenges, and lessons learned, to make it easier for you to replicate or adapt this project.

## Step-by-Step Guide

### 1. Importing Libraries
The first step in any ML project is to import the necessary libraries. At the outset, I wasn't sure which models or metrics I'd end up using—this is common in exploratory projects. You often start broad and refine as you go. I experimented with various options but settled on tools for data handling, visualization, modeling, and hyperparameter tuning.

Key libraries used include:
- **Pandas** and **NumPy** for data manipulation.
- **Scikit-learn** for preprocessing, metrics, and basic modeling.
- **XGBoost** for the gradient boosting model.
- **Optuna** for hyperparameter optimization.
- **Matplotlib/Seaborn** for visualizations.

Here's a screenshot from the final cleaned code showing the imports:

<img width="831" height="448" alt="Screenshot of imported libraries" src="https://github.com/user-attachments/assets/0d04af8f-1082-4d67-b1c5-42294e176846" />

**Learning Tip**: Always import libraries as you need them to avoid clutter. If you're new, start with a Jupyter notebook to test imports interactively. Remember, you can't predict everything upfront—iteration is key!

### 2. Loading and Exploring the Dataset
I sourced the dataset from Kaggle, which provided a clean starting point with features like episode details and popularity scores. However, real-world datasets are rarely perfect. For your own projects, I recommend searching multiple sources (e.g., Kaggle, UCI ML Repository, or domain-specific APIs) and intentionally working with "dirty" data to build robust skills. This dataset had about [X rows and Y columns—add specifics if known], with the target variable being listening time in minutes.

Initial exploration revealed:
- No major missing values, but some outliers in popularity scores.
- Categorical features (e.g., Genre, Publication_Day) needed encoding.
- Numerical features like Episode_Length_minutes showed a strong correlation with the target.

Screenshot of the dataset loading and initial inspection:

<img width="856" height="417" alt="Screenshot of dataset exploration" src="https://github.com/user-attachments/assets/f2c716be-f364-45b5-94a4-73da2d6e5bdc" />

**Learning Tip**: Use `df.info()`, `df.describe()`, and `df.head()` to get a quick overview. Visualize distributions with histograms to spot issues like skewness. Always check for imbalances in categorical data, as they can bias your model.

### 3. Data Cleaning and Preprocessing
Data cleaning is often 80% of the work in ML projects—don't skip it! After loading, I filtered out irrelevant columns (e.g., any metadata without predictive value) and handled missing or invalid entries. I applied Principal Component Analysis (PCA) to reduce dimensionality if needed, though the dataset wasn't overly complex. Feature importance analysis (using XGBoost's built-in tools) helped identify which variables mattered most—e.g., Episode_Length_minutes and Host_Popularity_percentage were top contributors.

Steps taken:
- Removed duplicates and handled outliers (e.g., capping extreme popularity scores).
- Encoded categorical variables (one-hot or label encoding).
- Scaled numerical features for model compatibility.
- Split the data into train/test sets (80/20 ratio).

Screenshot of the cleaning process, including PCA and feature importance:

<img width="850" height="502" alt="Screenshot of data cleaning and feature importance" src="https://github.com/user-attachments/assets/38c796ee-d8d7-4be3-98b5-d5f0aa0c41d4" />

**Learning Tip**: PCA is great for visualizing high-dimensional data or reducing noise, but use it judiciously—over-reduction can lose important information. Plot feature importance bar charts to guide decisions. Common pitfalls: Forgetting to handle categorical data or ignoring class imbalances.

### 4. Model Selection and Hyperparameter Tuning
I chose XGBoost, a powerful gradient boosting algorithm, because it handles regression tasks well, especially with mixed feature types, and is robust to outliers. To optimize it, I integrated Optuna, a hyperparameter tuning library that automates trial-and-error efficiently (e.g., tuning learning rate, max depth, and n_estimators).

Why this combo? XGBoost excels in structured data like this, and Optuna saves time compared to grid search. I ran 50-100 trials to find the best parameters.

Screenshot of model setup with Optuna and XGBoost:

<img width="837" height="475" alt="Screenshot of Optuna and XGBoost integration" src="https://github.com/user-attachments/assets/004c0020-630c-4919-87c4-d8a7a852578c" />

**Learning Tip**: Start with baseline models (e.g., linear regression) before advanced ones like XGBoost. Optuna uses Bayesian optimization, which is smarter than random search. Monitor for overfitting by comparing train vs. validation scores.

### 5. Training, Testing, and Evaluation
With the model tuned, I trained it on the preprocessed data and evaluated using cross-validation. Metrics focused on regression: MAE for average error, MSE for penalizing large errors, and R² for explained variance. The final model generalized well, with low errors on unseen data.

I also visualized predictions vs. actuals to spot patterns (e.g., underprediction for long episodes).

Screenshot of training, testing, and results:

<img width="851" height="573" alt="Screenshot of model training and evaluation" src="https://github.com/user-attachments/assets/5f66fb1f-4baa-424a-9f1c-07d476f7a654" />

**Learning Tip**: Use k-fold cross-validation (e.g., k=5) for reliable estimates. Plot residual errors to diagnose issues. If R² is high but MAE is off, check for heteroscedasticity (varying error magnitudes).

## Results and Insights
- **MAE**: 12.96 minutes (impressive for variable listening times).
- **MSE**: 13.03
- **R²**: ~0.98 (model explains nearly all variance).

This model could be extended for recommendation systems, e.g., suggesting episodes based on predicted engagement.

## Advice and Best Practices
Here are key lessons from this project to help you succeed in your own ML endeavors:

1. **Prioritize Data Quality**: Spend most of your time here—clean, explore, and understand your data. Garbage in, garbage out! Use tools like Pandas Profiling for automated insights.

2. **Deeply Understand Your Dataset**: Don't just load it; ask questions like "What does each feature mean?" and "Are there biases?" Domain knowledge (e.g., podcast trends) can reveal hidden patterns.

3. **Reevaluate Feature Importance**: Some features may seem minor in initial analysis but shine in training/testing. Always iterate: Remove low-importance ones and retest.

4. **Embrace Iteration and Experimentation**: ML is trial-and-error. Try multiple models (e.g., Random Forest, Neural Nets) and tuning methods. Track experiments with tools like MLflow.

5. **Bonus Tip: Version Control Everything**: Use Git for code and DVC for data/models. Document assumptions and failures—they're valuable learning opportunities.

6. **Ethical Considerations**: In prediction models like this, ensure fairness (e.g., no genre bias) and validate on diverse data.

This project is a solid foundation for regression tasks. Feel free to fork the repo, run the code, and tweak it. If you have questions, open an issue on GitHub! Happy modeling!
