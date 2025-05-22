# Emotion Classification Model - Test Plan

This document outlines the testing strategy for the emotion classification model, focusing on ensuring the end-to-end pipeline functions correctly after recent refactoring, using a dedicated test dataset.

## Test Data Setup (`tests/test_data/`)

*   **CSVs:** `tests/test_data/dummy_test_sent_emo.csv` (contains 1 dialogue, 2 utterances for 'test' split, corresponding to `Dialogue_ID: 110`).
*   **Videos:** `tests/test_data/test_videos/` contains `dia110_utt0.mp4` and `dia110_utt1.mp4`.
*   **Audio Output:** WAV files converted from the above MP4s will be saved to `tests/test_data/audio/test/`.
*   **HF Dataset Output:** Processed Hugging Face datasets will be saved to `tests/test_data/datasets/` (e.g., `tests/test_data/datasets/raw_wav/test/`).
*   **Configuration:** All tests will use `tests/test_config.yaml`, which is configured to use these paths and small data limits.

## Running the Automated Test Pipeline

The `tests/test_pipeline.py` script automates the execution of the core testing stages. 

1.  **Prerequisites:**
    *   Activate your Python virtual environment.
    *   Ensure all dependencies from `requirements.txt` are installed.
    *   Ensure `ffmpeg` and `ffprobe` are installed and available in your system PATH.

2.  **Execution:**
    ```bash
    python tests/test_pipeline.py
    ```

3.  **Manual Checkpoint Update (After Stage 3):
    *   After Stage 3 (Model Training Smoke Test) completes, the script will print a message indicating that training is done and where to find the checkpoint (typically in a subdirectory of `models/meld/mlp_fusion/checkpoints/` based on `test_config.yaml`).
    *   You **MUST** manually update the `TEST_CHECKPOINT_PATH` variable at the top of the `tests/test_pipeline.py` script with the full path to the actual `.ckpt` file that was generated.
    *   Re-run `python tests/test_pipeline.py`. Stages 1-3 might re-run (which is fine as they use `--force_conversion` or similar flags or are idempotent), but now Stage 4 (Evaluation) and Stage 5 (Inference) will use the correct checkpoint.

4.  **Cleanup (Optional):**
    *   The `tests/test_pipeline.py` script includes a `clean_previous_outputs()` function. You can uncomment the call to this function at the beginning of the `if __name__ == "__main__":` block to automatically clear generated test files (WAVs, HF datasets, inference results) before each full pipeline run.

## I. Data Acquisition & Initial Setup (Completed)

-   **Action:** The necessary test CSV (`dummy_test_sent_emo.csv` updated for Dialogue 110) and MP4 video files (`dia110_utt0.mp4`, `dia110_utt1.mp4`) are in place within `tests/test_data/` and `tests/test_data/test_videos/` respectively.
-   **Verification:** `tests/test_config.yaml` has been updated to point `raw_data_dir` to `tests/test_data`, `processed_audio_output_dir` to `tests/test_data/audio`, and `processed_hf_dataset_dir` to `tests/test_data/datasets`.

## II. WAV File Extraction (from MP4s) - Automated by `test_pipeline.py`

-   **Goal:** Verify that MP4s from `tests/test_data/test_videos/` are correctly converted to WAV format and saved into `tests/test_data/audio/test/`.
-   **Trigger:** Running `python tests/test_pipeline.py` executes Stage 1.
-   **Verification:**
    -   Target WAV files (e.g., `dia110_utt0.wav`, `dia110_utt1.wav`) are created in `tests/test_data/audio/test/`.
    -   Console output from the script shows no errors for Stage 1.
    -   Review summary statistics printed by the underlying script if available in logs.

## III. Hugging Face Dataset Preparation - Automated by `test_pipeline.py`

-   **Goal:** Verify creation of the processed Hugging Face dataset in `tests/test_data/datasets/` from the test WAVs and CSV.
-   **Trigger:** Running `python tests/test_pipeline.py` executes Stage 2.
-   **Verification:**
    -   Processed Hugging Face dataset is saved to `tests/test_data/datasets/raw_wav/test/`.
    -   Console output from the script shows no errors for Stage 2.
    -   Manually load the dataset using `datasets.load_from_disk("tests/test_data/datasets/raw_wav/test/")` in a separate Python session to inspect.
    -   Verify the number of items (should be 2 utterances).

## IV. Model Training (Smoke Test / Fast Dev Run) - Automated by `test_pipeline.py`

-   **Goal:** Ensure the training pipeline runs with the test HF dataset and produces a checkpoint.
-   **Trigger:** Running `python tests/test_pipeline.py` executes Stage 3.
-   **Verification:**
    -   Console output shows training progress for 1 epoch.
    -   A model checkpoint (`.ckpt` file) is saved (typically under `models/meld/mlp_fusion/checkpoints/` based on `test_config.yaml`). Note the path of this checkpoint.
    -   No crashes.

## V. Model Evaluation (Using Smoke Test Checkpoint) - Automated by `test_pipeline.py` (after checkpoint update)

-   **Goal:** Verify evaluation with the checkpoint from Stage IV.
-   **Trigger:** After updating `TEST_CHECKPOINT_PATH` in `tests/test_pipeline.py`, re-running the script executes Stage 4.
-   **Verification:**
    -   Console output shows evaluation metrics.
    -   No crashes.

## VI. Inference (Using Smoke Test Checkpoint) - Automated by `test_pipeline.py` (after checkpoint update)

-   **Goal:** Verify inference with the checkpoint from Stage IV.
-   **Trigger:** After updating `TEST_CHECKPOINT_PATH` in `tests/test_pipeline.py`, re-running the script executes Stage 5.
-   **Verification:**
    -   Console output shows predictions for the single audio file and CSV.
    -   `tests/test_data/inference_results.csv` is created and contains plausible predictions.
    -   No crashes.

## General Testing Tips

*   **Test Data Location:** All test inputs are in `tests/test_data/`, and outputs are configured via `tests/test_config.yaml` to also reside within `tests/test_data/` subdirectories (audio, datasets) or standard model save locations for checkpoints.
*   **Incremental Testing:** The `tests/test_pipeline.py` script runs stages sequentially. You can comment out later stages in the script if you want to focus on debugging an earlier stage.
*   **Configuration Management:** `tests/test_config.yaml` is the primary source of truth for test parameters.
*   **Environment:** Ensure `ffmpeg`/`ffprobe` are installed and all Python packages from `requirements.txt` are available.
---
This plan outlines using `tests/test_pipeline.py` for an automated run of the core test stages. 