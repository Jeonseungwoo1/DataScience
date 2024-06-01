import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Load data
df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# 1. Handle missing values
print("\n ==== Before ==== ")
print(df.isnull().sum())
print(df.info())

# 1-1. Remove rows with missing values in 'Name' column
df = df.dropna(subset=['Name'])

# 1-2. Replace 'tbd' in 'User_Score' with NaN and convert to float
df['User_Score'] = df['User_Score'].replace('tbd', np.nan)
df['User_Score'] = df['User_Score'].astype(float)

numerical_features = ['Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Score', 'User_Score']
categorical_features = ['Platform', 'Genre', 'Publisher']

# 1-3. Fill missing values in numerical features with median
imputer_num = SimpleImputer(strategy='median')
df[numerical_features] = imputer_num.fit_transform(df[numerical_features])

# 1-4. Fill missing values in categorical features with most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

print("\n ==== After ==== ")
print(df.isnull().sum())
print(df.info())

# 2. Scale numerical data
# Separate skewed and normal features
skewed_features = df[numerical_features].skew()[df[numerical_features].skew() > 1].index.tolist()
normal_features = [feature for feature in numerical_features if feature not in skewed_features]

standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

# Apply RobustScaler to skewed features
# Apply StandardScaler to normally distributed features
df[skewed_features] = robust_scaler.fit_transform(df[skewed_features])
df[normal_features] = standard_scaler.fit_transform(df[normal_features])

# 3. Encode categorical data
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
encoded_cat = encoder.fit_transform(df[categorical_features])
encoded_cat_df = pd.DataFrame(encoded_cat.toarray(), columns=encoder.get_feature_names_out(categorical_features))

# Combine encoded dataframe with the original dataframe
df = df.drop(columns=categorical_features)
df = pd.concat([df, encoded_cat_df], axis=1)

# 4. Prepare the dataset
df['Success'] = (df['Global_Sales'] > df['Global_Sales'].median()).astype(int)

X = df.drop(columns=['Success', 'Name', 'Global_Sales'])
y = df['Success']

# Check and remove remaining NaNs
print(X.isnull().sum())
X = X.dropna()
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation function
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    # K-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X_train, y_train, cv=kf, return_train_score=True, scoring=['accuracy', 'roc_auc'])

    print(f"\n{model_name} Cross-Validation Results:")
    print(f"Mean Train Accuracy: {np.mean(cv_results['train_accuracy']):.2f}")
    print(f"Mean Validation Accuracy: {np.mean(cv_results['test_accuracy']):.2f}")
    print(f"Mean Train ROC AUC: {np.mean(cv_results['train_roc_auc']):.2f}")
    print(f"Mean Validation ROC AUC: {np.mean(cv_results['test_roc_auc']):.2f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Evaluation
    print(f"\n{model_name} Test Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # ROC Curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()
        
    # Feature Importance (for Random Forest)
    if model_name == "Random Forest":
        importances = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        # Display the top 20 most important features
        top_n = 20
        top_features = feature_importances.head(top_n)
        
        print("\nTop 20 Feature Importances:")
        print(top_features)
        
        # Plotting the top 20 feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_features, y=top_features.index)
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()

# Set up model pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), encoded_cat_df.columns)
    ])

pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DecisionTreeClassifier(random_state=42))])

pipeline_knn = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier())])

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

# Evaluate models
evaluate_model(pipeline_dt, "Decision Tree", X_train, y_train, X_test, y_test)
evaluate_model(pipeline_knn, "K-Nearest Neighbors", X_train, y_train, X_test, y_test)
evaluate_model(pipeline_rf, "Random Forest", X_train, y_train, X_test, y_test)
