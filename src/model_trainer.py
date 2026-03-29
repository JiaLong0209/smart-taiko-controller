import pandas as pd
import numpy as np
import joblib
import os
import copy  # Needed to copy weights
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import Config

# === PyTorch Libraries ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
#  SETTINGS
# ==========================================
TRAIN_LIST = Config.SUPPORTED_MODELS 

# ==========================================
#  MODEL STRUCTURE (Restored to "Better" Version)
# ==========================================
class TaikoNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TaikoNet, self).__init__()
        # Architecture: 4 -> 64 -> 32 -> 16 -> 5
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.output(out)
        return out

# ==========================================
#  UTILS
# ==========================================
def get_threshold_prediction(row):
    dL, dR, kL, kR = row
    if max(row) < Config.TRIGGER_THRESHOLD: return 'Noise'
    max_don, max_ka = max(dL, dR), max(kL, kR)
    
    if max_don > max_ka * Config.CROSSTALK_RATIO:
        return 'Don_Left' if dL > dR else 'Don_Right'
    elif max_ka > max_don * Config.CROSSTALK_RATIO:
        return 'Ka_Left' if kL > kR else 'Ka_Right'
    else:
        return Config.SENSOR_LABELS[np.argmax(row)]

def simplify_label(label):
    if 'Don' in label: return 'Don'
    elif 'Ka' in label: return 'Ka'
    return 'Noise'

def plot_confusion_matrix(y_true, y_pred, labels, filename, model_name, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix ({model_name.upper()}) {title_suffix}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_comparison_chart(results, filename, title):
    # Sort results by value (accuracy) in ascending order for better looking bar charts
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
    models = [m.upper() for m in sorted_results.keys()]
    scores = list(sorted_results.values())
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(models)])
    plt.ylim(0, 110)
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.savefig(filename)
    plt.close()
    print(f"📊 Chart saved to {filename}")

def save_training_curves(history, filename):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['test_loss'], 'r--', label='Test Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['test_acc'], 'r--', label='Test Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==========================================
#  CORE TRAINING LOGIC
# ==========================================

def train_single_model(model_name, data_pack):
    X_train, X_test, y_train_enc, y_test_enc, X_train_scaled, X_test_scaled, le = data_pack
    
    print(f"\n{'='*40}")
    print(f"🚀 Training: {model_name.upper()}")
    print(f"{'='*40}")

    model_save_path = f'../models/taiko_{model_name}_model.pkl'
    if model_name == 'taikonet':
        model_save_path = f'../models/taiko_{model_name}_model.pth'

    y_test_labels = le.inverse_transform(y_test_enc)
    y_pred_labels = []

    # --- 1. Threshold ---
    if model_name == 'threshold':
        for row in X_test:
            pred = get_threshold_prediction(row)
            y_pred_labels.append(pred)
        y_pred_labels = np.array(y_pred_labels)

    # --- 2. PyTorch (TaikoNet) ---
    elif model_name == 'taikonet':
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.LongTensor(y_train_enc)
        X_test_t = torch.FloatTensor(X_test_scaled) 
        y_test_t = torch.LongTensor(y_test_enc)
        
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Using the deeper architecture
        model = TaikoNet(input_size=4, num_classes=len(Config.CLASSES))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.TAIKONET_LR)
        
        print("🔥 Training PyTorch Model (Saving BEST Accuracy)...")
        epochs = Config.TAIKONET_EPOCHS
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        
        # === BEST MODEL TRACKING ===
        best_acc = 0.0
        best_weights = copy.deepcopy(model.state_dict())

        with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                train_loss = running_loss / len(loader)
                train_acc = 100 * correct / total
                
                # === VALIDATION ===
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_t)
                    t_loss = criterion(test_outputs, y_test_t)
                    _, t_predicted = torch.max(test_outputs.data, 1)
                    t_correct = (t_predicted == y_test_t).sum().item()
                    
                    test_loss = t_loss.item()
                    test_acc = 100 * t_correct / y_test_t.size(0)

                # History
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)

                pbar.set_postfix({'Val_Acc': f'{test_acc:.1f}%', 'Best': f'{best_acc:.1f}%'})
                pbar.update(1)

                # === SAVE IF BEST ===
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_weights = copy.deepcopy(model.state_dict())
                    # Overwrite file immediately so taiko_main always gets the best
                    torch.save(best_weights, model_save_path) 
        
        print(f"💾 Saved BEST model (Acc: {best_acc:.2f}%) to {model_save_path}")
        save_training_curves(history, '../models/train_curves_taikonet.png')
        
        # Load Best for metrics
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_labels = le.inverse_transform(predicted.numpy())

    # --- 3. Traditional ML ---
    else:
        if model_name == 'svm':
            X_tr, X_te = X_train_scaled, X_test_scaled
            clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        elif model_name == 'rf':
            X_tr, X_te = X_train, X_test
            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        elif model_name == 'xgb':
            X_tr, X_te = X_train, X_test
            clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                eval_metric='mlogloss', use_label_encoder=False)
        
        clf.fit(X_tr, y_train_enc)
        y_pred_enc = clf.predict(X_te)
        y_pred_labels = le.inverse_transform(y_pred_enc)
        joblib.dump(clf, model_save_path)
        print(f"💾 Saved to {model_save_path}")

    # Metrics
    acc_full = accuracy_score(y_test_labels, y_pred_labels) * 100
    y_test_simple = [simplify_label(l) for l in y_test_labels]
    y_pred_simple = [simplify_label(l) for l in y_pred_labels]
    acc_simple = accuracy_score(y_test_simple, y_pred_simple) * 100
    
    print(f"🏆 Results [{model_name.upper()}] -> Full: {acc_full:.2f}% | Simple: {acc_simple:.2f}%")
    
    plot_confusion_matrix(y_test_labels, y_pred_labels, Config.CLASSES, 
                          f'../models/confusion_matrix_{model_name}_full.png', model_name, "- Full")
    plot_confusion_matrix(y_test_simple, y_pred_simple, ['Don', 'Ka', 'Noise'], 
                          f'../models/confusion_matrix_{model_name}_simple.png', model_name, "- Simple")
    
    return acc_full, acc_simple

def main():
    data_path = Config.CSV_PATH
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print(f"📂 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X = df[Config.SENSOR_LABELS].values
    y = df['Label'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    os.makedirs('../models', exist_ok=True)
    joblib.dump(le, '../models/label_encoder.pkl')

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, '../models/scaler.pkl') 

    data_pack = (X_train, X_test, y_train_enc, y_test_enc, X_train_scaled, X_test_scaled, le)
    results_full = {}
    results_simple = {}

    for model_name in TRAIN_LIST:
        acc_f, acc_s = train_single_model(model_name, data_pack)
        results_full[model_name] = acc_f
        results_simple[model_name] = acc_s

    print("\n" + "="*40)
    print("📊 GENERATING COMPARISON CHARTS")
    print("="*40)
    
    if len(TRAIN_LIST) > 0:
        save_comparison_chart(results_full, '../models/model_comparison_full.png', "Full Accuracy")
        save_comparison_chart(results_simple, '../models/model_comparison_simple.png', "Simplified Accuracy")
        
        # Sort results by full accuracy for printing (Ascending)
        sorted_m = sorted(results_full.keys(), key=lambda x: results_full[x])
        
        print(f"\n{'Model':<15} | {'Full Acc':<10} | {'Simple Acc':<10}")
        print("-" * 45)
        for m in sorted_m:
            print(f"{m.upper():<15} | {results_full[m]:.2f}%      | {results_simple[m]:.2f}%")

if __name__ == "__main__":
    main()