# 📝 Training Run Report

**Run ID:** `{{ run_id }}`  
**Date/Time:** `{{ timestamp }}`  
**Experiment Name:** `{{ experiment_name }}`  
**Model Architecture:** `{{ model_architecture }}`  
**Dataset:** `{{ dataset_name }}` (Split used for training: `{{ train_split }}`, Preprocessing details: `{{ preprocess_details }}`)  
**Associated YAML Config:** `{{ path_to_yaml_config }}`  
**Git Commit Hash:** `N/A (Not Tracked)`

## 1. Configuration Summary

*   **Model Specifics:**
    *   Text Encoder: `{{ text_encoder_model_name }}` (Frozen: `{{ freeze_text_encoder }}`)
    *   Audio Encoder: `{{ audio_encoder_model_name }}` (Frozen: `{{ freeze_audio_encoder }}`)
    *   Other Key Model Params: `{{ other_model_params_summary }}`
*   **Hyperparameters:**
    *   Learning Rate: `{{ learning_rate }}`
    *   Batch Size (Train): `{{ batch_size_train }}` (Eval: `{{ batch_size_eval }}`)
    *   Optimizer: `{{ optimizer_name }}` (Weight Decay: `{{ weight_decay }}`)
    *   LR Scheduler: `{{ lr_scheduler_name }}`
    *   Max Epochs: `{{ num_epochs }}` (Actual Epochs Trained: `{{ actual_epochs_trained }}`)
    *   Early Stopping Patience: `{{ early_stopping_patience }}` (Triggered: `{{ early_stopping_triggered }}`)
    *   Gradient Accumulation Steps: `{{ gradient_accumulation_steps }}`
*   **Environment:**
    *   Python Version: `{{ python_version }}`
    *   Key Libraries: `pytorch: {{ pytorch_version }}`, `transformers: {{ transformers_version }}`, `pytorch-lightning: {{ lightning_version }}`, `datasets: {{ datasets_version }}` (Full list: `{{ requirements_file_path_or_dump }}`)
    *   Hardware: `{{ hardware_specs }}` (CPU details. GPU: `{{ gpu_name }}`, Total GPU Memory: `{{ gpu_total_memory_gb }}`)
*   **Random Seed:** `{{ random_seed }}`
*   **Dataset Slicing (if any for this run):**
    *   `limit_dialogues_train: {{ limit_dialogues_train }}`
    *   `limit_dialogues_dev: {{ limit_dialogues_dev }}`
    *   `limit_dialogues_test: {{ limit_dialogues_test }}`

## 2. Performance Metrics

| Metric               | Train      | Validation (`{{ eval_split }}`) | Test (`{{ test_split }}`) |
|----------------------|------------|-----------------------|--------------------|
| Accuracy             | `{{ acc_train }}` | `{{ acc_val }}`         | `{{ acc_test }}`     |
| Precision (macro)    | `{{ prec_train }}`| `{{ prec_val }}`        | `{{ prec_test }}`    |
| Recall (macro)       | `{{ rec_train }}` | `{{ rec_val }}`         | `{{ rec_test }}`     |
| F₁ Score (macro)     | `{{ f1_train }}`  | `{{ f1_val }}`          | `{{ f1_test }}`      |
| F₁ Score (weighted)  | `{{ wf1_train }}` | `{{ wf1_val }}`         | `{{ wf1_test }}`     |
| ROC-AUC (macro OVR)  | `{{ auc_train }}` | `{{ auc_val }}`         | `{{ auc_test }}`     |
<!-- Add other relevant metrics, e.g., per-class F1 scores -->

**Per-Class F1 Scores (Validation - `{{ eval_split }}`):**
```
{{ per_class_f1_val_json_or_table }}
```

**Per-Class F1 Scores (Test - `{{ test_split }}`):**
```
{{ per_class_f1_test_json_or_table }}
```

## 3. Loss Curves & Visualizations

*   **Final Training Loss:** `{{ final_train_loss }}`
*   **Best Validation Loss:** `{{ best_val_loss }}` (Epoch: `{{ best_val_loss_epoch }}`)
*   **Plots (Paths relative to this report or absolute if hosted):**
    *   Training & Validation Loss Curves: `![Loss Curve]({{ loss_curve_path }})`
    *   Validation Confusion Matrix: `![Validation CM]({{ val_confusion_matrix_path }})`
    *   Test Confusion Matrix: `![Test CM]({{ test_confusion_matrix_path }})`
    *   (Optional) ROC Curves: `![ROC Curve]({{ roc_curve_path }})`

## 4. Efficiency & Resource Usage

*   **Total Training Time:** `{{ total_training_time }}` (HH:MM:SS)
*   **Average Time per Epoch:** `{{ avg_time_per_epoch }}` (MM:SS)
*   **GPU Utilization (Avg/Peak):** `{{ gpu_utilization_avg_peak }}` (e.g., from W&B or nvidia-smi logs)
*   **GPU Memory Usage (Avg/Peak):** `{{ gpu_memory_avg_peak }}`
*   **CPU Utilization (Avg/Peak):** `{{ cpu_utilization_avg_peak }}`
*   **System Peak RAM Usage:** `{{ system_ram_peak }}`
*   **Model Checkpoint Size:** `{{ checkpoint_size_mb }}` MB

## 5. Notes, Observations & Next Steps

*   **Key Observations During Training:**
    *   `{{ observation_1 }}`
    *   `{{ observation_2 }}`
*   **Comparison to Baseline/Previous Runs (if applicable):**
    *   `{{ comparison_notes }}`
*   **Potential Issues or Areas for Improvement:**
    *   `{{ issue_1 }}`
*   **Actionable Next Steps / Future Experiments:**
    1.  `{{ next_step_1 }}`
    2.  `{{ next_step_2 }}`

---
*This report was generated for Run ID: `{{ run_id }}`.* 