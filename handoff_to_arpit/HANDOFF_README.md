# Suraksha-Net: Model Deployment Handoff Package
**Prepared by:** Vishnu Bhargava (Advanced DL Model Development & Evaluation Lead)  
**Target:** Arpit / Deployment & Optimization Team  
**Model Architecture:** CNN_LSTM (ResNet18 Backbone + LSTM Temporal Aggregation)  

---

## 1. Quick Start (Single Tested Command)

Run inference on any real CCTV video clip with one command:

```bash
# 1. Run inference on a video (automatically utilizes CUDA if available):
python run_inference.py --video path/to/cctv_clip.mp4

# 2. Specify custom alert threshold or device:
python run_inference.py --video path/to/cctv_clip.mp4 --threshold 0.30 --device cuda
```

### Example Real Output:
```json
{
  "status": "success",
  "video": "/absolute/path/to/cctv_clip.mp4",
  "prediction": "Violence",
  "violence_probability": 0.9424,
  "non_violence_probability": 0.0576,
  "alert_triggered": true,
  "alert_threshold": 0.30,
  "device_used": "cuda"
}
```

---

## 2. Input Specifications & Preprocessing Pipeline
- **Input Tensor Format:** `[B, 16, 3, 224, 224]` (Batch × Sequence × Channels × Height × Width)
- **Sequence Length:** 16 frames uniformly sampled across video duration.
- **Spatial Dimensions:** 224 × 224 pixels.
- **Color Space:** RGB (`cv2.COLOR_BGR2RGB`).
- **Normalization:** ImageNet statistics:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Scaling:** Pixels scaled to `[0.0, 1.0]` before normalization.

---

## 3. Class Mapping & Decision Threshold
- **Class Mapping:**
  - `0`: **Non-Violence** (Normal human activity, walking, loitering, conversation)
  - `1`: **Violence** (Physical fighting, assault, aggressive altercations)
- **Calibrated Alert Threshold:** **`0.30`**
  - Tuned on the validation set to maximize violence sensitivity (recall).
  - An event triggers an active security alert whenever `P(Violence) >= 0.30`.

---

## 4. Final Benchmark Performance Metrics (Held-Out Test Set)

Evaluated on untouched test videos (100 multi-camera surveillance clips):

| Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **Accuracy** | **83.00%** | Overall correct classifications |
| **Precision** | **82.81%** | Reliability of raised alarms |
| **Violence Recall (Sensitivity)** | **89.83%** | Catch rate for real violent incidents |
| **F1-Score** | **86.18%** | Harmonic balance between alarm purity and detection |
| **ROC-AUC** | **0.8876** | Discriminative power across all thresholds |
| **False Alarms (FP)** | 11 clips | Non-violent clips flagged due to rapid motion/running |
| **Missed Fights (FN)** | 6 clips | Subtle scuffles or low-contrast night conditions |

---

## 5. File Manifest
- `best_model.pth`: PyTorch state dictionary and architecture configuration (`137 MB`)
- `run_inference.py`: Production-ready inference runner (zero extra external dependencies)
- `model_config.json`: Full architecture hyperparameters (ResNet18, 512 feat dim, 256 LSTM hidden, 1 layer)
- `preprocessing_config.json`: Normalization mean/std, tensor dimensions, and frame sampling parameters
- `class_mapping.json`: Integer to string class mapping dictionary
- `recommended_threshold.json`: Calibrated validation alert threshold and security justification
- `VISHNU_WORK_COMPREHENSIVE_REPORT.pdf`: Complete 12-page defense report and experimental documentation
