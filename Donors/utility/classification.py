# Use non-GUI backend for matplotlib in Django environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import os

# ================= ADDED =================
import shap
from sklearn.metrics import classification_report
# ========================================

warnings.filterwarnings('ignore')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU detected: {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected. Running on CPU.")

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print(f"Original shape: {df.shape}")
    
    df['selected'] = df['procured'].astype(bool).astype(int)

    drop_cols = [
        'OPO', 'PatientID', 'HospitalID',
        'time_asystole', 'time_brain_death', 'time_referred', 'time_approached',
        'time_authorized', 'time_procured',
        'outcome_heart', 'outcome_liver', 'outcome_kidney_left', 'outcome_kidney_right',
        'outcome_lung_left', 'outcome_lung_right', 'outcome_intestine', 'outcome_pancreas',
        'procured', 'transplanted', 'authorized'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Fill numeric columns with median, handling cases where median is NaN
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        median_val = df[col].median()
        if pd.isna(median_val):
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(median_val, inplace=True)
    
    # Fill categorical/object columns with mode, handling cases where all values are NaN
    for col in df.select_dtypes(include=['object', 'bool']).columns:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col].fillna(mode_val[0], inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)

    df['referral_to_approach'] = np.where(df.get('approached', False) == True, 'within_24h', 'not_approached')

    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    selected_features = [
        'Age', 'Referral_Year', 'Procured_Year', 'brain_death_True',
        'Mechanism_of_Death_ICH/Stroke', 'ABO_BloodType_A1', 'ABO_BloodType_O',
        'approached_True', 'Eye_Referral_True', 'referral_to_approach_within_24h',
        'selected'
    ]
    df = df[[col for col in selected_features if col in df.columns]]
    
    # Final NaN check and removal
    df = df.dropna()

    print(f"Shape after preprocessing: {df.shape}")
    return df

def prepare_data(df, target_column='selected'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Remove any remaining NaN values
    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_smote = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Final NaN check
    if np.isnan(X_train_smote).any() or np.isnan(X_test).any():
        print("Warning: NaN values found after scaling. Replacing with 0.")
        X_train_smote = np.nan_to_num(X_train_smote, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

    return X_train_smote, X_test, y_train, y_test, scaler

def initialize_models(input_shape):
    return {
        'KNN': KNeighborsClassifier(),
        'LR': LogisticRegression(max_iter=1000),
        'RF': RandomForestClassifier(random_state=42),
        'NB': GaussianNB(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'ANN': Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    }

def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test):
    if name == 'ANN':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        y_pred_proba = model.predict(X_test).flatten()
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'Model': name,
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_pred_proba)
    }, model, recall_score(y_test, y_pred)

def save_model(name, model):
    os.makedirs('models', exist_ok=True)
    if name == 'ANN':
        model.save(f'models/{name}_model.h5')
    else:
        joblib.dump(model, f'models/{name}_model.pkl')

# ================= ADDED =================
def generate_shap(best_xgb, best_lr, X_train, X_test, feature_names):

    os.makedirs('static/plots', exist_ok=True)

    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # ===================== (a) XGBOOST — KERNEL EXPLAINER =====================
    background = shap.sample(X_train_df, 100, random_state=42)

    expl_xgb = shap.KernelExplainer(
        best_xgb.predict_proba,
        background
    )

    # IMPORTANT: call explainer like a function
    shap_values_xgb = expl_xgb(X_test_df.iloc[:100])

    # Beeswarm
    shap.summary_plot(
        shap_values_xgb.values[:, :, 1],  # class-1 explanations
        X_test_df.iloc[:100],
        show=False
    )
    plt.savefig('static/plots/fig1a_beeswarm.png', bbox_inches='tight')
    plt.close()

    # Bar
    shap.summary_plot(
        shap_values_xgb.values[:, :, 1],
        X_test_df.iloc[:100],
        plot_type='bar',
        show=False
    )
    plt.savefig('static/plots/fig1a_bar.png', bbox_inches='tight')
    plt.close()

    # ===================== (b) LOGISTIC REGRESSION =====================
    expl_lr = shap.LinearExplainer(best_lr, X_train_df)
    shap_values_lr = expl_lr(X_test_df)

    shap.summary_plot(
        shap_values_lr.values,
        X_test_df,
        show=False
    )
    plt.savefig('static/plots/fig1b_beeswarm.png', bbox_inches='tight')
    plt.close()

    shap.summary_plot(
        shap_values_lr.values,
        X_test_df,
        plot_type='bar',
        show=False
    )
    plt.savefig('static/plots/fig1b_bar.png', bbox_inches='tight')
    plt.close()


# ========================================

from django.conf import settings

def classification_View():
    # Load and preprocess data
    file_path = os.path.join(settings.MEDIA_ROOT, 'referrals.csv')
    data = load_data(file_path)
    data = preprocess_data(data)
    
    # ================= ADDED =================
    # Extract feature names BEFORE scaling (needed for SHAP)
    feature_names = data.drop(columns=['selected']).columns.tolist()
    # ========================================
    
    # Prepare data
    target_column = 'selected'
    X_train_smote, X_test, y_train_smote, y_test, scaler = prepare_data(data, target_column)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Initialize models
    models = initialize_models(X_train_smote.shape[1])
    
    # Train and evaluate models
    results = []
    best_model = None
    best_recall = 0
    best_model_name = ''

    # ================= ADDED =================
    best_xgb = None
    best_lr = None
    # ========================================
    
    for name, model in models.items():
        result, trained_model, recall = train_and_evaluate_model(
            name, model,
            X_train_smote, y_train_smote,
            X_test, y_test
        )
        results.append(result)
        save_model(name, trained_model)
        
        # ================= ADDED =================
        # Track XGBoost and Logistic Regression for SHAP
        if name == 'XGBoost':
            best_xgb = trained_model
        if name == 'LR':
            best_lr = trained_model
        # ========================================
        
        # Track best model based on recall
        if recall > best_recall:
            best_recall = recall
            best_model = trained_model
            best_model_name = name
    

    # ================= ADDED =================
    # Generate SHAP plots (Fig 1a & 1b)
    # This SAVES the plots correctly for Django
    try:
        if best_xgb is not None and best_lr is not None:
            generate_shap(
                best_xgb,
                best_lr,
                X_train_smote,
                X_test,
                feature_names
            )
    except Exception as e:
        print(f"Error generating SHAP plots: {e}")
        # Continue without SHAP plots
    # ========================================
    

    # Convert results to DataFrame (in-memory)
    import pandas as pd
    results_df = pd.DataFrame(results)

    # Prepare data for template rendering
    results_table = results_df.to_dict(orient='records')

    context = {
        'results_table': results_table,
        'best_model': best_model_name,
        'best_recall': round(best_recall, 4),

        # ================= ADDED =================
        # SHAP figure paths (for HTML <img>)
        'fig1a_beeswarm': 'plots/fig1a_beeswarm.png',
        'fig1a_bar': 'plots/fig1a_bar.png',
        'fig1b_beeswarm': 'plots/fig1b_beeswarm.png',
        'fig1b_bar': 'plots/fig1b_bar.png',
        # ========================================
    }

    return context


def get_donor_input(feature_names):
   
    donor_data = []
    for feature in feature_names:
        while True:
            try:
                donor_data.append(feature)
                break
            except ValueError:
                print(f"Please enter a valid number for {feature}.")
    return np.array([donor_data])

def predict_with_best_model(best_model_name, best_model, scaler,  feature_names):
    donor_input = get_donor_input(feature_names) 
    donor_input = np.array(donor_input).reshape(1, -1)

    # Now scale full input
    donor_input_scaled = scaler.transform(donor_input)

    # Then apply feature selection
    
    if best_model_name == 'ANN':
        prediction = (best_model.predict(donor_input_scaled) > 0.5).astype(int).flatten()
    else:
        prediction = best_model.predict(donor_input_scaled)

    return 'Selected' if prediction[0] == 1 else 'Not Selected'


