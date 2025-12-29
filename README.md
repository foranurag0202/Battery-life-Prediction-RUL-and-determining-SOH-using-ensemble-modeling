# Battery Life Prediction: Remaining Useful Life (RUL) and State of Health (SOH) using Ensemble Modeling

This repository presents an ensemble deep learning framework for lithium-ion battery health prognosis, focusing on accurate **capacity degradation modeling**, **State of Health (SOH) estimation**, and **Remaining Useful Life (RUL) prediction**.

The project integrates multiple state-of-the-art time-series forecasting models and combines their predictions using an attention-based ensemble approach to improve robustness and degradation trend consistency.

---

## Project Overview

Accurate battery health prediction is critical for applications such as electric vehicles, energy storage systems, and predictive maintenance. This work addresses the limitations of single-model forecasting by leveraging an ensemble of heterogeneous architectures capable of capturing both short-term dynamics and long-term degradation trends.

The framework performs autoregressive capacity prediction over battery discharge cycles and derives SOH and EOL metrics based on capacity threshold analysis.

---

## Models Used

The following base models are implemented and trained independently:

- Transformer
- Autoformer
- DLinear
- XLSTM

Predictions from these base models are combined using an **LSTM with Attention-based Ensemble Model**, which learns optimal weighting across models for improved forecasting stability.

All trained base models and the trained ensemble model are included in this repository.

---

## Dataset

The experiments are conducted using the **NASA Lithium-Ion Battery Dataset**, which contains charge–discharge cycle data for multiple batteries operated under controlled conditions.

Included in the repository:
- Preprocessed battery capacity data
- Cycle-wise discharge information
- Datasets used for training and evaluation

---

## Prediction Strategy

- Sliding window sequence modeling with a fixed window size of 16 cycles
- Autoregressive one-step-ahead prediction
- Iterative forecasting up to the last available cycle in the dataset
- SOH estimation derived from predicted capacity values
- EOL detection based on capacity threshold crossing

The current implementation predicts within the available dataset horizon and does not extrapolate beyond unseen future cycles.

---

## Evaluation Metrics

Model performance is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)
- End-of-Life (EOL) Prediction Error (in cycles)

It is observed that even with low capacity prediction error and high R² values, EOL prediction error may remain significant due to the threshold-based nature of EOL detection. This behavior is consistent with findings reported in battery prognostics literature.

---

## Results

- The ensemble model achieves strong capacity prediction accuracy with low MAE and RMSE.
- High R² values indicate effective learning of degradation trends.
- Output graphs comparing predicted and actual capacity degradation curves are included in the repository.
- EOL prediction error highlights sensitivity near the degradation knee region.

---

## Repository Contents

- Trained base model weights
- Trained ensemble model
- Dataset files
- Output graphs and visualizations
- Source code for model training, evaluation, and plotting

---
## Output Graphs



## Limitations

- EOL is derived indirectly via capacity threshold crossing rather than direct RUL regression.
- Predictions are limited to the dataset time horizon.
- No uncertainty quantification is currently implemented.

---

## Future Work

- Direct RUL prediction using multi-task learning
- Knee-point-aware loss functions
- Uncertainty-aware forecasting
- Cross-dataset generalization using additional battery datasets

---

## Usage

Clone the repository and run the main script to reproduce predictions and plots. Ensure that required dependencies are installed as specified in the code.

---

## Author

Anurag Kumar  
B.Tech, Electronics and Communication Engineering  
National Institute of Technlogy Andhra Pradesh

---

## License

This project is intended for academic and research purposes.
