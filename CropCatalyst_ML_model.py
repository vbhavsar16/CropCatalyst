import warnings

# Data Manipulation
import pandas as pd
import numpy as np

# Imputation - RandomForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import imgkit

# Data Transformation
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler

# Feature Selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV

# Pipeline
from sklearn.pipeline import Pipeline

# Metrics
import math
from sklearn.metrics import mean_squared_error, make_scorer

# Regression Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')


# ------------------- Getting the dataset that we prepared ------------------#
df_combined = pd.read_csv("RebootTheEarth/Datasets/Final_dataset.csv")

print(df_combined.describe(include='all'))

df_na = df_combined.isna().sum().reset_index(name="missing_values")
df_na["percentage"] = round((df_na["missing_values"] / df_combined.shape[0]) * 100, 2)
print(df_na.sort_values(by="percentage", ascending=False)[:])


# ------------------- Visualizing skewness in yield data ------------------#
plt.figure(figsize=(10, 6))
sns.histplot(df_combined["Yield"], kde=True)
plt.grid(True)
plt.xlabel('Yield (tons/hectare)')
plt.savefig('Skewness_yield.jpg', bbox_inches='tight', dpi=600)

print("Skewness (Yield): ", round(df_combined["Yield"].skew(), 2))

# ------------------- Removing the skewness from yield and visualizing ------------------#
df_combined["Yield"] = np.log1p(df_combined["Yield"])

plt.figure(figsize=(10, 6))
sns.histplot(df_combined["Yield"], kde=True)
plt.grid(True)
plt.xlabel('Yield (tons/hectare)')
plt.savefig('Skewness_yield_log_transformed.jpg', bbox_inches='tight', dpi=600)


# ------------------- Visualizing and removing the skewness from other data ------------------#
print("Skewness (Yield): ", round(df_combined["Yield"].skew(), 2))
df_numeric = df_combined.drop(columns=["longitude", "latitude", "crop", "season", "soil type", "Yield"], axis=1)

num_plots = len(df_numeric.columns)
num_columns = 2
num_rows = num_plots // num_columns + (1 if num_plots % num_columns > 0 else 0)

plt.figure(figsize=(20, 6 * num_rows))

for i, column in enumerate(df_numeric.columns):
    plt.subplot(num_rows, num_columns, i + 1)
    sns.histplot(df_numeric[column], kde=True)
    plt.title(f'{column} Distribution')

plt.tight_layout()
plt.show()


skewness = df_numeric.apply(lambda x: x.skew()).sort_values(ascending=False)
skewed = skewness[abs(skewness) > 0.5]
skewed_features = skewed.index

for i in skewed_features:
    df_combined[i] = np.log1p(df_combined[i])

df_v2 = df_combined.copy()

# ------------------- Changing to one-hot coding the crop and season data ------------------#
df_v2['crop'] = df_v2['crop'].replace({'maize': 1, 'rice': 2, 'soybean': 3, 'wheat': 4})
df_v2['season'] = df_v2['season'].replace({'major': 1, 'second': 2, 'spring': 3, 'winter': 4})

df_object = df_combined.select_dtypes(include=["object"])

df_v2 = df_v2.dropna()

X, X_test, y, y_test = train_test_split(df_v2, df_v2['Yield'], test_size=0.3, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_cv = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1], max_iter=1000, cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)

selected_features = X.columns[(lasso_cv.coef_ == 0).ravel()].tolist()
#print(len(selected_features))
#print(selected_features)

postlasso_df = df_v2[selected_features + ["Yield"]]

data = postlasso_df.copy()

# ------------------- Getting the train and test data splits ------------------#
traindataset = data[data["Yield"].notnull()]

# train and test split
X = traindataset.drop(columns=["Yield"], axis=1)
y = traindataset["Yield"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# ------------------- Defining the models ------------------#
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'RandomForestRegressor': RandomForestRegressor(),
    'XGBRegressor': XGBRegressor(objective='reg:squarederror'),
    'LGBMRegressor': LGBMRegressor(verbose=-1),
    'CatBoostRegressor': CatBoostRegressor(verbose=0)
}

params = {
    'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
    'Lasso': {'model__alpha': [0.001, 0.01, 0.1]},
    'ElasticNet': {'model__alpha': [0.001, 0.01, 0.1], 'model__l1_ratio': [0.2, 0.5, 0.8]},
    'SVR': {'model__C': [0.1, 1, 10], 'model__epsilon': [0.1, 0.2, 0.5], 'model__kernel': ['linear', 'rbf']},
    'RandomForestRegressor': {'model__n_estimators': [10, 50, 100]},
    'XGBRegressor': {'model__learning_rate': [0.1, 0.5], 'model__max_depth': [3, 5, 7]},
    'LGBMRegressor': {'model__learning_rate': [0.1, 0.5], 'model__num_leaves': [31, 50, 100]},
    'CatBoostRegressor': {'model__learning_rate': [0.01, 0.1, 0.5], 'model__depth': [4, 6, 8]}
}

results = []


# ------------------- Defining the accuracy matric ------------------#
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


# ------------------- Training and Testing the models ------------------#
for name, model in models.items():
    pipeline = Pipeline([
        ('model', model)
    ])
    if name in params:
        grid_search = GridSearchCV(pipeline, params[name], cv=5, scoring=rmse_scorer)
    else:
        grid_search = GridSearchCV(pipeline, {}, cv=5, scoring=rmse_scorer)
        
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    r2 = grid_search.best_estimator_.score(X_test, y_test)
    rmse_val = rmse(y_test, y_pred)
    results.append({
        'Model': name,
        'R-Squared Score': r2,
        'RMSE Score': rmse_val,
        'Best Parameters': grid_search.best_params_
    })

results_df = pd.DataFrame(results)

# ------------------- Visualizing the results ------------------#
pd.set_option('display.max_colwidth', None)
print(results_df)


html = results_df.to_html()
imgkit.from_string(html, 'Results.jpg')