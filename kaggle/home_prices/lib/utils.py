import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

def plot_feature_distribution(df, feature_name):
    """
    Plot the distribution of a given feature.
    
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - feature_name (str): The name of the feature/column to plot.
    """
    sns.histplot(df[feature_name], kde=True)
    plt.title(f'Distribution of {feature_name}')
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plot a heatmap of feature correlations.
    
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    """
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()


def feature_importance(df, target_column):
    """
    Return feature importance using a RandomForest.
    
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - target_column (str): The name of the target/output column.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.show()

def plot_learning_curve(estimator, X, y, cv=None):
    """
    Plot learning curves for an estimator.
    
    Parameters:
    - estimator (estimator object): An estimator instance implementing 'fit'.
    - X (array-like): Input data.
    - y (array-like): Target values.
    - cv (int or cross-validation generator, default=None): Determines the cross-validation splitting strategy.
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def hyperparameter_tuning(estimator, param_grid, X, y, cv=None):
    """
    Perform Grid Search for hyperparameters.
    
    Parameters:
    - estimator (estimator object): An estimator instance implementing 'fit'.
    - param_grid (dict or list of dictionaries): Dictionary with parameters names as keys and lists of parameter settings to try as values.
    - X (array-like): Input data.
    - y (array-like): Target values.
    - cv (int or cross-validation generator, default=None): Determines the cross-validation splitting strategy.
    """
    grid_search = GridSearchCV(estimator, param_grid, cv=cv)
    grid_search.fit(X, y)
    return grid_search.best_params_


def plot_residuals(y_true, y_pred):
    """
    Plot residuals to analyze errors.
    
    Parameters:
    - y_true (array-like): True target values.
    - y_pred (array-like): Predicted target values by the model.
    """
    residuals = y_true - y_pred
    sns.histplot(residuals)
    plt.title('Residuals Distribution')
    plt.show()


def analyze_feature_distributions(df, threshold_skewness=0.5):
    """
    Analyze distributions of all features in the dataframe.
    
    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - threshold_skewness (float, optional): Threshold for considering a feature as skewed. Default is 0.5.
    
    Returns:
    - summary (pandas.DataFrame): A dataframe containing summary statistics for each feature.
    """
    summary = []

    for column in df.select_dtypes(include=['number']).columns:
        # Compute basic statistics
        mean = df[column].mean()
        median = df[column].median()
        std_dev = df[column].std()
        skew = df[column].skew()

        # Identify skewness type
        if abs(skew) <= threshold_skewness:
            skewness_type = 'Symmetric'
        elif skew > threshold_skewness:
            skewness_type = 'Right Skewed'
        else:
            skewness_type = 'Left Skewed'
        
        # Plot the distribution
        # sns.histplot(df[column], kde=True)
        # plt.title(f'Distribution of {column}')
        # plt.show()

        # Append statistics to summary list
        summary.append([column, mean, median, std_dev, skewness_type])

    # Convert summary list to DataFrame
    summary_df = pd.DataFrame(summary, columns=['Feature', 'Mean', 'Median', 'Std Dev', 'Skewness Type'])
    
    return summary_df

def preprocessing_recommendations(summary_df, skew_threshold=0.5, high_std_dev_multiplier=2.0):
    """
    Analyze the summary statistics of feature distributions and provide preprocessing recommendations with code snippets.
    
    Parameters:
    - summary_df (pandas.DataFrame): A dataframe containing summary statistics for each feature.
    - skew_threshold (float, optional): Threshold for considering a feature as skewed. Default is 0.5.
    - high_std_dev_multiplier (float, optional): Multiplier of the median standard deviation to consider a feature's variability as high. Default is 2.0.
    
    Returns:
    - recommendations (dict): A dictionary with feature names as keys and lists of recommendations with code snippets as values.
    - grouped_features (dict): A dictionary grouping features by the type of problem.
    """
    
    recommendations = {}
    grouped_features = {
        'rightSkewed': [],
        'leftSkewed': [],
        'highVariability': []
    }
    
    median_std_dev = summary_df['Std Dev'].median()
    
    for _, row in summary_df.iterrows():
        feature = row['Feature']
        recommendations[feature] = []

        # Check for skewness
        if row['Skewness Type'] == 'Right Skewed':
            recommendations[feature].append({
                'recommendation': "Consider applying a logarithmic or square root transformation to reduce right skewness.",
                'code': f"df['{feature}'] = np.log1p(df['{feature}'])  # Log transformation\n"
                        f"df['{feature}'] = np.sqrt(df['{feature}'])  # Square root transformation"
            })
            grouped_features['rightSkewed'].append(feature)
        elif row['Skewness Type'] == 'Left Skewed':
            recommendations[feature].append({
                'recommendation': "Consider applying a power or exponential transformation to reduce left skewness.",
                'code': f"df['{feature}'] = df['{feature}'] ** 2  # Power transformation\n"
                        f"df['{feature}'] = np.exp(df['{feature}'])  # Exponential transformation"
            })
            grouped_features['leftSkewed'].append(feature)

        # Check for high variability
        if row['Std Dev'] > high_std_dev_multiplier * median_std_dev:
            recommendations[feature].append({
                'recommendation': "High variability detected. Consider scaling the feature or investigate for potential outliers.",
                'code': f"from sklearn.preprocessing import StandardScaler\n"
                        f"scaler = StandardScaler()\n"
                        f"df['{feature}'] = scaler.fit_transform(df[['{feature}']])  # Scaling the feature"
            })
            grouped_features['highVariability'].append(feature)
        
        # If no recommendations for the feature, state it
        if not recommendations[feature]:
            recommendations[feature].append({
                'recommendation': "No specific preprocessing recommendations.",
                'code': "# No specific code required."
            })

    return recommendations, grouped_features
