import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import LSTM
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import load_data # This now imports from the dataset.py file
import os

# --- 1. Load Data ---
# Ensure this path points to the directory containing the .mat files
dataset_path = "D:/machine learning/03_206/Code/" 

print("Loading dataset...")
Battery = load_data(dataset_path)
print("Dataset loaded successfully!")

# --- 2. Model Definitions ---

### Common: PositionalEncoding ###
class PositionalEncoding(nn.Module):
    def __init__(self, feature_len, feature_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(feature_len, feature_size)
        position = torch.arange(0, feature_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_size, 2).float() * (-math.log(10000.0) / feature_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

### Model A: TransformerNet ###
class TransformerNet(nn.Module):
    def __init__(self, feature_size=2, seq_len=16, d_model=128, nhead=8, num_layers=4, dropout=0.0, use_pos_encoding=True):
        super(TransformerNet, self).__init__()
        self.input_layer = nn.Linear(feature_size, d_model)
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoder = PositionalEncoding(seq_len, d_model, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, 1)
        self._init_weights()

    def forward(self, x):
        x = self.input_layer(x)
        if self.use_pos_encoding: x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.output_layer(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

### Model B: AutoReformer ###
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model):
        super(AutoCorrelationLayer, self).__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q, K, V = self.query_proj(x), self.key_proj(x), self.value_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return self.output_proj(context)

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super(AutoformerEncoderLayer, self).__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.auto_corr = AutoCorrelationLayer(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = self.auto_corr(seasonal)
        seasonal = self.ff(seasonal)
        return seasonal + trend

class AutoformerNet(nn.Module):
    def __init__(self, feature_size=1, seq_len=16, d_model=256, num_layers=2, kernel_size=3):
        super(AutoformerNet, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        self.pos_encoding = PositionalEncoding(feature_len=seq_len, feature_size=d_model)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model=d_model, kernel_size=kernel_size) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers: x = layer(x)
        out = x.mean(dim=1)
        return self.output_layer(out)

### Model C: DLinear ###
class AdvancedDLinear(nn.Module):
    def __init__(self, feature_size=1, seq_len=16, hidden_size=256, num_layers=4, dropout_rate=0.2):
        super(AdvancedDLinear, self).__init__()
        self.input_dim = seq_len * feature_size
        trend_layers = []
        for _ in range(num_layers - 1):
            trend_layers.extend([
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate),
            ])
        self.trend_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate),
            *trend_layers, nn.Linear(hidden_size, 1)
        )
        season_layers = []
        for _ in range(num_layers - 1):
            season_layers.extend([
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate),
            ])
        self.season_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate),
            *season_layers, nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        trend = self.trend_layer(x)
        season = self.season_layer(x - trend)
        return trend + season

### Model D: XLSTM ###
class XLSTMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(XLSTMBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.xlstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        x_lstm, _ = self.xlstm(x_norm)
        x = x + self.dropout1(x_lstm)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x

class XLSTMNet(nn.Module):
    def __init__(self, feature_size=1, seq_len=16, d_model=256, num_layers=2, dropout=0.1, use_pos_encoding=True):
        super(XLSTMNet, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(seq_len, d_model, dropout)
        self.encoder = nn.Sequential(*[
            XLSTMBlock(d_model=d_model, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, 1)
        self._init_weights()

    def forward(self, x):
        x = self.input_proj(x)
        if self.use_pos_encoding: x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.output_layer(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

# --- 3. Ensembling Model and Helpers ---

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context).squeeze(-1)

class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

def generate_autoregressive_output(model, Battery, target_battery_id, window_size, pred_steps, device):
    model.eval()
    with torch.no_grad():
        capacities = Battery[target_battery_id][1]
        capacities_tensor = torch.tensor(capacities, dtype=torch.float32)
        input_seq = capacities_tensor[:window_size].clone()
        sequence = input_seq.tolist()
        for _ in range(pred_steps):
            x_input = input_seq[-window_size:].unsqueeze(0).unsqueeze(-1)  # [1, 16, 1]
            if model.__class__.__name__ == 'TransformerNet':
                time_feature = torch.arange(window_size).float().unsqueeze(0).unsqueeze(-1)  # [1, 16, 1]
                x_input = torch.cat([x_input, time_feature], dim=-1)  # [1, 16, 2]
            x_input = x_input.to(device)
            pred = model(x_input)
            pred_value = pred.item()
            sequence.append(pred_value)
            input_seq = torch.cat([input_seq, torch.tensor([pred_value])])
    return np.array(sequence)

def train_lstm_ensemble(X_all, y_all, epochs=300, lr=0.001):
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    model = LSTMWithAttention(input_size=X.shape[-1])
    criterion = LogCoshLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Ensemble Training Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    return model

def lstm_ensemble_forecast(model, model_outputs, window_size):
    X = np.stack(model_outputs, axis=-1)
    time_index = np.linspace(0, 1, len(model_outputs[0]))
    X = np.concatenate([X, time_index[:, None]], axis=-1)
    X_seq, _ = create_sequences(X, np.zeros(len(X)), window_size)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    return np.concatenate([np.zeros(window_size), preds])

# ============================
# Evaluation Metrics for Ensemble Model
# ============================

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(true_seq, pred_seq, window_size):
    """
    true_seq : actual capacity values (numpy array)
    pred_seq : predicted capacity values (numpy array from ensemble)
    """

    # Align lengths
    y_true = true_seq[window_size:]
    y_pred = pred_seq[window_size:]

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    # Optional: End-of-life prediction error (capacity hits 70% of initial)
    initial_capacity = y_true[0]
    eol_threshold = 0.8 * initial_capacity

    try:
        true_eol = np.where(y_true <= eol_threshold)[0][0]
        pred_eol = np.where(y_pred <= eol_threshold)[0][0]
        eol_error = abs(pred_eol - true_eol)
    except IndexError:
        eol_error = None  # If model never predicts EOL

    # Print evaluation results
    print("\n===== Ensemble Model Evaluation =====")
    print(f"MAE  : {mae:.5f}")
    print(f"RMSE : {rmse:.5f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.5f}")
    if eol_error is not None:
        print(f"EOL Prediction Error: {eol_error} cycles")
    else:
        print("EOL not reached by prediction.")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "EOL_Error": eol_error
    }





# --- 4. Main Execution ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Models ---
    # Ensure these .pth files are in the same directory as the script
    model_paths = {
        "AutoReformer": os.path.join(dataset_path, "AutoReformer.pth"),
        "Transformer": os.path.join(dataset_path, "transformer_model.pth"),
        "DLinear": os.path.join(dataset_path, "DLinear.pth"),
        "XLSTM": os.path.join(dataset_path, "XLSTM.pth")
    }

    model_AutoReformer = AutoformerNet(feature_size=1, seq_len=16, d_model=256, num_layers=2, kernel_size=3).to(device)
    model_Transformer = TransformerNet(feature_size=2, seq_len=16, d_model=128, num_layers=4, dropout=0.0).to(device)
    model_Dlinear = AdvancedDLinear(feature_size=1, seq_len=16, hidden_size=256, num_layers=4, dropout_rate=0.3).to(device)
    model_xLSTM = XLSTMNet(feature_size=1, seq_len=16, d_model=256, num_layers=2, dropout=0.0).to(device)

    model_AutoReformer.load_state_dict(torch.load(model_paths["AutoReformer"], map_location=device))
    model_Transformer.load_state_dict(torch.load(model_paths["Transformer"], map_location=device))
    model_Dlinear.load_state_dict(torch.load(model_paths["DLinear"], map_location=device))
    model_xLSTM.load_state_dict(torch.load(model_paths["XLSTM"], map_location=device))

    models = {
        "AutoReformer": model_AutoReformer,
        "Transformer": model_Transformer,
        "DLinear": model_Dlinear,
        "XLSTM": model_xLSTM
    }

    # --- Prepare Data for Ensemble Training ---
    battery_ids = [  'B0005','B0007', 'B0018']
    X_all, y_all = [], []
    window_size = 16

    print("\nGenerating base model predictions for ensemble training...")
    for battery_id in battery_ids:
        model_outputs = []
        for model_name, model in models.items():
            pred_steps = len(Battery[battery_id][1]) - window_size
            predictions = generate_autoregressive_output(model, Battery, battery_id, window_size, pred_steps, device)
            model_outputs.append(predictions)

        time_index = np.linspace(0, 1, len(model_outputs[0]))
        X = np.stack(model_outputs + [time_index], axis=-1)
        y = np.array(Battery[battery_id][1][:len(model_outputs[0])])
        
        X_seq, y_seq = create_sequences(X, y, window_size)
        X_all.append(X_seq)
        y_all.append(y_seq)

    # --- Train Ensemble Model ---
    print("\nTraining the LSTM+Attention ensemble model...")
    shared_model = train_lstm_ensemble(X_all, y_all)
    print("Ensemble model training complete.")



    # # --- Evaluate and Plot ---
    # print("\nGenerating final plots...")
    # fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    # axs = axs.flatten()

    # for i, battery_id in enumerate(battery_ids):
    #     pred_steps = len(Battery[battery_id][1]) - window_size
    #     model_outputs = [
    #         generate_autoregressive_output(models[name], Battery, battery_id, window_size, pred_steps, device)
    #         for name in models
    #     ]

    #     target_values = np.array(Battery[battery_id][1][:len(model_outputs[0])])
    #     final_predictions = lstm_ensemble_forecast(shared_model, model_outputs, window_size)

    #     axs[i].plot(target_values[window_size:], label='Original Sequence')
    #     axs[i].plot(final_predictions[window_size:], label='Ensemble Prediction', linestyle='--')
    #     axs[i].set_title(f'Battery {battery_id}')
    #     axs[i].set_xlabel('Time Steps (Cycles)')
    #     axs[i].set_ylabel('Capacity (Ah)')
    #     axs[i].legend()
    #     axs[i].grid(True)

    # plt.tight_layout()
    # plt.suptitle("Ensemble Model Forecast vs. Actual", fontsize=16, y=1.02)
    # plt.savefig(os.path.join(dataset_path, "ensemble_forecastx1.png"))
    # print("Plot saved as ensemble_forecast.png")





    # --- Generating the Final Plot for B0005 Only ---
    print("\nGenerating final plot for B0006...")

    # Define the specific battery you want to plot
    battery_id = 'B0006' 
    window_size = 16 # Make sure window_size is defined

    # Create a single plot instead of a 2x2 grid
    fig, ax = plt.subplots(1, 1, figsize=(10, 8)) 

    # --- Prepare data for the single plot ---
    pred_steps = len(Battery[battery_id][1]) - window_size
    model_outputs = [
        generate_autoregressive_output(models[name], Battery, battery_id, window_size, pred_steps, device)
        for name in models
    ]

    target_values = np.array(Battery[battery_id][1][:len(model_outputs[0])])
    final_predictions = lstm_ensemble_forecast(shared_model, model_outputs, window_size)

        # ---- RUN EVALUATION FOR B0006 (or any battery) ----
    metrics = evaluate_predictions(target_values, final_predictions, window_size)

    # --- Plot the data on the single axis 'ax' ---
    ax.plot(target_values[window_size:], label='Original Sequence')
    ax.plot(final_predictions[window_size:], label='Ensemble Prediction', linestyle='--')
    ax.set_title(f'Ensemble Model Forecast vs. Actual for Battery {battery_id}', fontsize=16)
    ax.set_xlabel('Time Steps (Cycles)')
    ax.set_ylabel('Capacity (Ah)')
    ax.legend()
    ax.grid(True)

    # --- Finalize and save the plot ---
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path, "ensemble_forecast_B0006_1wu06.png"))
    print("Plot saved as ensemble_forecast_B0005_01.png")
    plt.show()



