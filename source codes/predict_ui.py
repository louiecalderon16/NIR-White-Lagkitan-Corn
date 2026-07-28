#!/usr/bin/env python3
import csv
import os
import subprocess
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

PREDICTION_IMPORT_ERROR = None

try:
    import joblib
    import numpy as np
    import qwiic_as7265x
except ImportError as exc:
    PREDICTION_IMPORT_ERROR = str(exc)


SCAN_COUNT = 10
SCAN_DELAY_SECONDS = 3
LED_SETTLE_SECONDS = 0.05
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "final_model.joblib"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
HISTORY_FILENAME = "prediction_history.csv"
HISTORY_PATH = os.path.join(BASE_DIR, HISTORY_FILENAME)
DEFAULT_BINARY_CLASS_LABELS = ["Not Sweet", "Sweet"]
PREDICTION_CONFIDENCE_THRESHOLD = 0.60

feature_cols = ['730nm','760nm','810nm','860nm','900nm','940nm']

def filter_scans_stage1(X, keep_ratio_fallback=0.85):
    # X is (n_scans, n_features) array
    centroid = X.mean(axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    threshold = med + 1.5 * mad
    mask = dists <= threshold
    if mask.sum() < len(X) * 0.5:
        cutoff = np.quantile(dists, keep_ratio_fallback)
        mask = dists <= cutoff
    return X[mask]

def apply_snv(X):
    # X is (n, features)
    snv_X = []
    for row in X:
        row = row.astype(float)
        mean = row.mean()
        std = row.std(ddof=1)
        if std < 1e-8:
            continue  # skip invalid rows
        snv_row = (row - mean) / std
        snv_X.append(snv_row)
    return np.array(snv_X)

def stage2_median_aggregation(X):
    # X is (n, features)
    return np.median(X, axis=0)

RAW_VALIDATION_CHANNEL_MIN = np.array([130.0, 105.0, 225.0, 1560.0, 74.0, 28.0], dtype=float)
RAW_VALIDATION_CHANNEL_MAX = np.array([390.0, 280.0, 610.0, 3800.0, 165.0, 66.0], dtype=float)
RAW_VALIDATION_SHAPE_REF = np.array([0.46866578, 0.32556862, 0.67333615, 4.22802605, 0.21442854, 0.08997486], dtype=float)
RAW_VALIDATION_SHAPE_THRESHOLD = 0.32
RAW_VALIDATION_PER_SCAN_SHAPE_THRESHOLD = 0.36
RAW_VALIDATION_MIN_SCAN_STD = 85.0
RAW_VALIDATION_MIN_MEDIAN_STD = 24.0
RAW_VALIDATION_SCAN_CONSISTENCY_THRESHOLD = 0.30
RAW_VALIDATION_MAX_BETWEEN_SCAN_DRIFT = 0.32


def validate_raw_scans(raw_scans):
    raw = np.asarray(raw_scans, dtype=float)

    if raw.size == 0:
        return False, "No valid sample detected. Please position the corn properly and try again."
    if raw.ndim != 2 or raw.shape[1] != len(feature_cols):
        return False, "No valid sample detected. Please position the corn properly and try again."
    if np.isnan(raw).any():
        return False, "No valid sample detected. Please position the corn properly and try again."
    if np.allclose(raw, raw[0], atol=1e-2):
        return False, "No valid sample detected. Please position the corn properly and try again."

    scan_std = np.std(raw, axis=1)
    if np.median(scan_std) < RAW_VALIDATION_MIN_SCAN_STD:
        return False, "No valid sample detected. Please position the corn properly and try again."

    median_spectrum = np.median(raw, axis=0)
    if np.any(median_spectrum < RAW_VALIDATION_CHANNEL_MIN) or np.any(median_spectrum > RAW_VALIDATION_CHANNEL_MAX):
        return False, "Not a corn sample. Please scan a valid corn ear."
    if np.std(median_spectrum) < RAW_VALIDATION_MIN_MEDIAN_STD:
        return False, "No valid sample detected. Please position the corn properly and try again."

    if abs(median_spectrum.mean()) < 1e-8:
        return False, "No valid sample detected. Please position the corn properly and try again."

    normalized_spectrum = median_spectrum / median_spectrum.mean()
    shape_distance = np.linalg.norm(normalized_spectrum - RAW_VALIDATION_SHAPE_REF)
    if shape_distance > RAW_VALIDATION_SHAPE_THRESHOLD:
        return False, "Not a corn sample. Please scan a valid corn ear."

    for i, scan in enumerate(raw):
        scan_mean = scan.mean()
        if abs(scan_mean) < 1e-8:
            return False, "No valid sample detected. Please position the corn properly and try again."
        scan_norm = scan / scan_mean
        per_scan_distance = np.linalg.norm(scan_norm - RAW_VALIDATION_SHAPE_REF)
        if per_scan_distance > RAW_VALIDATION_PER_SCAN_SHAPE_THRESHOLD:
            return False, "Not a corn sample. Please scan a valid corn ear."

    filtered = filter_scans_stage1(raw)
    if filtered.shape[0] < max(2, raw.shape[0] // 2):
        return False, "No valid sample detected. Please position the corn properly and try again."

    if raw.shape[0] >= 4:
        first_half = np.median(raw[:raw.shape[0]//2], axis=0)
        second_half = np.median(raw[raw.shape[0]//2:], axis=0)
        
        if abs(first_half.mean()) < 1e-8 or abs(second_half.mean()) < 1e-8:
            return False, "No valid sample detected. Please position the corn properly and try again."
        first_norm = first_half / first_half.mean()
        second_norm = second_half / second_half.mean()
        half_distance = np.linalg.norm(first_norm - second_norm)
        
        if half_distance > RAW_VALIDATION_SCAN_CONSISTENCY_THRESHOLD:
            return False, "Sample unstable. Please hold it steady and rescan."

    for i in range(raw.shape[0] - 1):
        scan_i_mean = raw[i].mean()
        scan_next_mean = raw[i+1].mean()
        if abs(scan_i_mean) < 1e-8 or abs(scan_next_mean) < 1e-8:
            return False, "No valid sample detected. Please position the corn properly and try again."
        scan_i_norm = raw[i] / scan_i_mean
        scan_next_norm = raw[i+1] / scan_next_mean
        consecutive_dist = np.linalg.norm(scan_i_norm - scan_next_norm)
        
        if consecutive_dist > RAW_VALIDATION_MAX_BETWEEN_SCAN_DRIFT:
            return False, "Sample unstable. Please hold it steady and rescan."

    return True, ""


class PredictionGUI:
    def __init__(self, root):
        self.root = root
        self.fullscreen = False
        self.sensor = None
        self.prediction_in_progress = False
        self.model = None
        self.class_labels = []
        self.status_var = tk.StringVar(value="Initializing prediction UI...")
        self.result_var = tk.StringVar(value="Prediction result: -")
        self.model_info_var = tk.StringVar(value="Model details: -")
        self.capture_info_var = tk.StringVar(value="Capture summary: -")
        self.progress_var = tk.DoubleVar(value=0)
        self.measurement_var = tk.StringVar(value="")
        self.countdown_var = tk.StringVar(value="")
        self.current_frame = None  # Track which frame is shown

        self.root.title("Corn Sweetness Prediction Assistant")
        try:
            self.root.state('zoomed')
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                self.root.geometry("900x700")
        self.root.bind("<Escape>", self.toggle_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)

        for column, weight in enumerate((1, 20, 1)):
            self.root.grid_columnconfigure(column, weight=weight, uniform="predict-cols")
        for row, weight in enumerate((0, 0, 0, 0, 0, 0, 0, 0, 0, 1)):
            self.root.grid_rowconfigure(row, weight=weight)

        # Create main prediction frame
        self.main_frame = ttk.Frame(self.root, padding=14)
        self.main_frame.grid(row=0, column=1, rowspan=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.main_frame, text="Corn Sweetness Estimator", font=("TkDefaultFont", 20, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        thesis_text = (
            "Thesis project using AS7265x sensor. Collects scans with rotation, applies filtering, SNV, median aggregation, then classifies."
        )
        ttk.Label(self.main_frame, text=thesis_text, wraplength=600, justify="left", font=("TkDefaultFont", 12)).grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(0, 14)
        )

        ttk.Label(self.main_frame, text="LED setup", font=("TkDefaultFont", 13, "bold")).grid(row=2, column=0, sticky="nw", padx=(0, 8), pady=(12, 4))
        self.led_frame = ttk.Frame(self.main_frame)
        self.led_frame.grid(row=2, column=1, columnspan=2, sticky="w", pady=(12, 4))
        self.led_white_var = tk.BooleanVar(value=True)
        self.led_ir_var = tk.BooleanVar(value=True)
        self.led_uv_var = tk.BooleanVar(value=True)
        self.led_checkbuttons = [
            tk.Checkbutton(self.led_frame, text="White LED", variable=self.led_white_var, font=("TkDefaultFont", 12), onvalue=True, offvalue=False),
            tk.Checkbutton(self.led_frame, text="IR LED", variable=self.led_ir_var, font=("TkDefaultFont", 12), onvalue=True, offvalue=False),
            tk.Checkbutton(self.led_frame, text="UV LED", variable=self.led_uv_var, font=("TkDefaultFont", 12), onvalue=True, offvalue=False),
        ]
        self.led_checkbuttons[0].grid(row=0, column=0, padx=(0, 12))
        self.led_checkbuttons[1].grid(row=0, column=1, padx=(0, 12))
        self.led_checkbuttons[2].grid(row=0, column=2, padx=(0, 12))

        # Instructions for manual rotation
        rotation_instructions = (
            "During scanning, manually rotate the corn sample between each measurement to capture spectral variation "
            "from different angles."
        )
        ttk.Label(self.main_frame, text=rotation_instructions, wraplength=600, justify="left", font=("TkDefaultFont", 10, "italic")).grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(0, 8)
        )

        # Button frame for Predict and View History
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(18, 8))
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=0)

        self.predict_button = ttk.Button(button_frame, text="Predict", command=self.start_prediction)
        self.predict_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.history_button = ttk.Button(button_frame, text="View History", command=self.show_history_frame)
        self.history_button.grid(row=0, column=1, sticky="ew", padx=(8, 8))

        self.shutdown_button = ttk.Button(button_frame, text="Shutdown", command=self.confirm_shutdown, width=12)
        self.shutdown_button.grid(row=0, column=2, sticky="e")

        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            orient="horizontal",
            mode="determinate",
            maximum=SCAN_COUNT,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(self.main_frame, textvariable=self.measurement_var, wraplength=600, justify="left", font=("TkDefaultFont", 12, "italic")).grid(
            row=6, column=0, columnspan=3, sticky="ew", pady=(0, 0)
        )
        ttk.Label(self.main_frame, textvariable=self.countdown_var, wraplength=600, justify="left", font=("TkDefaultFont", 14, "bold")).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(0, 0)
        )
        ttk.Label(self.main_frame, textvariable=self.result_var, wraplength=600, justify="left", font=("TkDefaultFont", 14, "bold")).grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=4
        )
        ttk.Label(self.main_frame, textvariable=self.status_var, wraplength=600, justify="left", font=("TkDefaultFont", 11)).grid(
            row=9, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )
        ttk.Label(self.main_frame, textvariable=self.model_info_var, wraplength=600, justify="left", font=("TkDefaultFont", 11)).grid(
            row=10, column=0, columnspan=3, sticky="ew", pady=4
        )

        # Create history viewer frame
        self._create_history_frame()
        self.history_frame.grid_remove()  # Hide initially

        self._load_model()
        self._init_sensor()

        if PREDICTION_IMPORT_ERROR is not None:
            self.status_var.set(f"Prediction dependencies are missing: {PREDICTION_IMPORT_ERROR}")
            self.predict_button.configure(state="disabled")
            for checkbutton in self.led_checkbuttons:
                checkbutton.configure(state="disabled")

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        try:
            self.root.attributes("-fullscreen", self.fullscreen)
        except tk.TclError:
            pass
        if not self.fullscreen:
            self.root.geometry("900x700")

    def _load_model(self):
        if not os.path.isfile(MODEL_PATH):
            self.model_info_var.set(f"Model details: {MODEL_FILENAME} not found at {MODEL_PATH}")
            self.predict_button.configure(state="disabled")
            return
        try:
            self.model = joblib.load(MODEL_PATH)
            raw_classes = getattr(self.model, "classes_", None)
            
            # Handle different class representations
            if raw_classes is not None:
                class_values = raw_classes.tolist() if hasattr(raw_classes, "tolist") else list(raw_classes)
                if class_values and all(isinstance(v, str) for v in class_values):
                    self.class_labels = [str(v) for v in class_values]
                elif class_values:
                    # Try numeric or default labels
                    if len(class_values) == 2:
                        self.class_labels = list(DEFAULT_BINARY_CLASS_LABELS)
                    else:
                        self.class_labels = [f"Class {i}" for i in range(len(class_values))]
                else:
                    self.class_labels = list(DEFAULT_BINARY_CLASS_LABELS)
            else:
                # Model doesn't have classes_ attribute, assume binary
                self.class_labels = list(DEFAULT_BINARY_CLASS_LABELS)
            
            # Try to get model type and test predict_proba availability
            model_type = type(self.model).__name__
            has_proba = hasattr(self.model, "predict_proba")
            classes = ", ".join(self.class_labels) or "unknown"
            
            self.model_info_var.set(
                f"Model: {model_type}. Classes: {classes}."
            )
            print(f"[DEBUG] Model loaded: {model_type}. Has predict_proba: {has_proba}. Classes: {self.class_labels}")
        except Exception as exc:
            self.model_info_var.set(f"Model error: unable to load {MODEL_FILENAME} ({str(exc)})")
            self.predict_button.configure(state="disabled")
            print(f"[DEBUG] Model loading error: {type(exc).__name__}: {exc}")

    def _init_sensor(self):
        try:
            self.sensor = qwiic_as7265x.QwiicAS7265x()
            if not self.sensor.is_connected():
                self.status_var.set("Sensor not connected. Check wiring and I2C.")
                self.predict_button.configure(state="disabled")
                return
            if not self.sensor.begin():
                self.status_var.set("Failed to initialize sensor.")
                self.predict_button.configure(state="disabled")
                return
            self.sensor.set_integration_cycles(50)
            self.sensor.set_gain(self.sensor.kGain64x)
            self.sensor.disable_indicator()
            self.status_var.set("Sensor ready. Press Predict to start.")
        except Exception as exc:
            self.status_var.set(f"Sensor error: {exc}")
            self.predict_button.configure(state="disabled")

    def _set_controls_state(self, state):
        self.predict_button.configure(state=state)
        self.shutdown_button.configure(state=state)
        for checkbutton in self.led_checkbuttons:
            checkbutton.configure(state=state)

    def confirm_shutdown(self):
        if self.prediction_in_progress:
            messagebox.showwarning("Shutdown Blocked", "Prediction is currently in progress. Please wait until it finishes.")
            return

        if not messagebox.askyesno("Confirm Shutdown", "Are you sure you want to shutdown the device?"):
            return

        self.status_var.set("Shutting down device...")
        self._set_controls_state("disabled")
        self.root.update_idletasks()
        try:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        except Exception as exc:
            messagebox.showerror("Shutdown Failed", f"Unable to shutdown device: {exc}")
            self.status_var.set("Shutdown failed. Please try again or use the physical power button after saving.")
            self._set_controls_state("normal")

    def _prepare_feature_vector(self, raw_scans):
        raw_scan_matrix = np.asarray(raw_scans, dtype=float)

        # Stage 1: Filter scans
        filtered_matrix = filter_scans_stage1(raw_scan_matrix)
        inlier_count = len(filtered_matrix)
        total_scans = len(raw_scan_matrix)
        inlier_text = f"Inlier scans kept: {inlier_count}/{total_scans}."

        # SNV
        snv_matrix = apply_snv(filtered_matrix)

        # Stage 2: Median aggregation
        feature_vector = stage2_median_aggregation(snv_matrix).tolist()

        # For history and display, use the processed vector
        processed_scan = feature_vector

        return feature_vector, processed_scan, inlier_text

    def _predict_label(self, feature_vector):
        x_values = np.asarray([feature_vector], dtype=float)
        predictions = self.model.predict(x_values)
        predicted_value = int(predictions[0]) if isinstance(predictions[0], (int, np.integer)) else predictions[0]
        
        # Get class label
        if isinstance(predicted_value, int) and 0 <= predicted_value < len(self.class_labels):
            predicted_label = self.class_labels[predicted_value]
        else:
            predicted_label = str(predicted_value)
        
        # Calculate confidence
        confidence_text = "Confidence: unavailable"
        try:
            if hasattr(self.model, "predict_proba"):
                # predict_proba expects 2D array (samples, features), so reshape
                probabilities = self.model.predict_proba(x_values.reshape(1, -1))[0]
                confidence_value = float(np.max(probabilities))
                confidence_text = f"Confidence: {confidence_value:.2%}"
                
                if confidence_value < PREDICTION_CONFIDENCE_THRESHOLD:
                    return "Uncertain", f"{confidence_text}. Best match: {predicted_label}"
            else:
                print(f"[DEBUG] Model {type(self.model).__name__} does not have predict_proba method")
        except Exception as e:
            print(f"[DEBUG] predict_proba error: {type(e).__name__}: {e}")
            print(f"[DEBUG] Model type: {type(self.model).__name__}")
            print(f"[DEBUG] Input shape: {x_values.shape}, dtype: {x_values.dtype}")
            confidence_text = f"Confidence: error ({type(e).__name__})"

        return predicted_label, confidence_text

    def _update_progress(self, completed_scans):
        self.progress_var.set(completed_scans)
        if completed_scans < SCAN_COUNT:
            self.measurement_var.set("")
            self.status_var.set(f"Scan {completed_scans}/{SCAN_COUNT} completed.")
            self._start_countdown(SCAN_DELAY_SECONDS)
        else:
            self.status_var.set(f"All {SCAN_COUNT} scans completed. Processing...")
            self.countdown_var.set("")
            self.measurement_var.set("")

    def _start_countdown(self, seconds):
        if seconds > 0:
            self.countdown_var.set(f"⚠ ROTATE THE SAMPLE NOW! Next scan in {seconds} second{'s' if seconds != 1 else ''} ⚠")
            self.root.after(1000, lambda: self._start_countdown(seconds - 1))
        else:
            self.countdown_var.set("")

    def _set_measurement_text(self, text):
        self.measurement_var.set(text)

    def _finish_prediction_success(self, predicted_label, confidence_text, capture_text, averaged_scan, led_flags):
        self.progress_var.set(0)
        self.countdown_var.set("")
        self.measurement_var.set("")
        self.prediction_in_progress = False
        self._set_controls_state("normal")
        self.result_var.set(f"Prediction result: {predicted_label}. {confidence_text}")
        self.status_var.set("Prediction completed.")
        # Save to history
        self._save_prediction_result(predicted_label, confidence_text, averaged_scan, led_flags)

    def _finish_prediction_error(self, error_message):
        self.progress_var.set(0)
        self.countdown_var.set("")
        self.measurement_var.set("")
        self.prediction_in_progress = False
        self._set_controls_state("normal")
        self.status_var.set("Prediction failed.")
        messagebox.showerror("Prediction Error", error_message)

    def _prediction_worker(self, enabled_devices, led_flags):
        try:
            raw_scans = []

            try:
                # Keep the LEDs on for the full scan sequence.
                for dev in enabled_devices:
                    self.sensor.enable_bulb(dev)
                time.sleep(LED_SETTLE_SECONDS)

                for scan_index in range(SCAN_COUNT):
                    self.root.after(0, self._set_measurement_text, f"Taking measurement {scan_index + 1}/{SCAN_COUNT}...")
                    self.sensor.take_measurements()
                    raw_scan = [
                        self.sensor.get_calibrated_t(),
                        self.sensor.get_calibrated_u(),
                        self.sensor.get_calibrated_v(),
                        self.sensor.get_calibrated_w(),
                        self.sensor.get_calibrated_k(),
                        self.sensor.get_calibrated_l(),
                    ]
                    raw_scans.append(raw_scan)
                    self.root.after(0, self._set_measurement_text, f"Measurement {scan_index + 1}/{SCAN_COUNT} complete.")
                    self.root.after(0, self._update_progress, scan_index + 1)
                    if scan_index < SCAN_COUNT - 1:
                        time.sleep(SCAN_DELAY_SECONDS)
            finally:
                # Ensure LEDs are off after completion
                for dev in enabled_devices:
                    self.sensor.disable_bulb(dev)

            try:
                scan_matrix = np.asarray(raw_scans, dtype=float)
                if scan_matrix.shape[0] > 1:
                    all_same = np.allclose(scan_matrix, scan_matrix[0], atol=1e-8)
                    unique_rows = np.unique(np.round(scan_matrix, 6), axis=0).shape[0]
                    channel_stats = [
                        (float(scan_matrix[:, idx].min()), float(scan_matrix[:, idx].max()), float(scan_matrix[:, idx].std()))
                        for idx in range(scan_matrix.shape[1])
                    ]
                    print(f"[DEBUG] scan_matrix shape={scan_matrix.shape}, all_same={all_same}, unique_rows={unique_rows}")
                    print(f"[DEBUG] per-channel stats (min,max,std): {channel_stats}")
            except Exception as e:
                print(f"[DEBUG] unable to compute scan diagnostics: {e}")

            valid, validation_message = validate_raw_scans(raw_scans)
            if not valid:
                self.root.after(0, self._finish_prediction_error, validation_message)
                return

            feature_vector, processed_scan, inlier_text = self._prepare_feature_vector(raw_scans)
            try:
                print(f"[DEBUG] processed_scan={['{:.6f}'.format(v) for v in processed_scan]}")
                print(f"[DEBUG] feature_vector={['{:.6f}'.format(v) for v in feature_vector]}")
            except Exception as e:
                print(f"[DEBUG] unable to print feature vector: {e}")
            predicted_label, confidence_text = self._predict_label(feature_vector)

            # Debug output
            try:
                probabilities = self.model.predict_proba(np.asarray([feature_vector], dtype=float))[0]
                print("Probabilities:", probabilities)
            except Exception as e:
                print("[DEBUG] Could not print probabilities:", e)

            # Prepare nm channel labels and values for display
            nm_labels = ["730nm", "760nm", "810nm", "860nm", "900nm", "940nm"]
            nm_str = ", ".join(f"{label}={value:.4f}" for label, value in zip(nm_labels, processed_scan))

            capture_text = (
                f"Capture summary: {SCAN_COUNT} scans acquired with manual rotation between each scan, stage1 filtered, SNV normalized, and median aggregated. "
                f"LED flags: Wh={led_flags[0]}, IR={led_flags[1]}, UV={led_flags[2]}.\n"
                f"Processed readings: {nm_str}\n{inlier_text}"
            )
            self.root.after(0, self._finish_prediction_success, predicted_label, confidence_text, capture_text, processed_scan, led_flags)
        except Exception as exc:
            self.root.after(0, self._finish_prediction_error, str(exc))

    def start_prediction(self):
        if self.prediction_in_progress:
            messagebox.showinfo("Prediction Running", "Please wait for the current prediction to finish.")
            return
        if PREDICTION_IMPORT_ERROR is not None:
            messagebox.showerror("Missing Dependency", f"Prediction dependencies are missing: {PREDICTION_IMPORT_ERROR}")
            return
        if self.sensor is None:
            messagebox.showwarning("No Sensor", "Sensor is not ready.")
            return
        if self.model is None:
            messagebox.showwarning("No Model", "Model failed to load.")
            return

        enabled_devices = []
        if self.led_white_var.get():
            enabled_devices.append(self.sensor.kLedWhite)
        if self.led_ir_var.get():
            enabled_devices.append(self.sensor.kLedIr)
        if self.led_uv_var.get():
            enabled_devices.append(self.sensor.kLedUv)
        led_flags = [1 if self.led_white_var.get() else 0, 1 if self.led_ir_var.get() else 0, 1 if self.led_uv_var.get() else 0]

        self.prediction_in_progress = True
        self.progress_var.set(0)
        self.countdown_var.set("")
        self.result_var.set(f"Prediction result: collecting {SCAN_COUNT} scans with manual rotation...")
        self.status_var.set("Starting multi-scanning sequence with manual rotation...")
        self._set_controls_state("disabled")
        threading.Thread(target=self._prediction_worker, args=(enabled_devices, led_flags), daemon=True).start()

    def _create_history_frame(self):
        """Create the history viewer frame with Treeview."""
        self.history_frame = ttk.Frame(self.root, padding=14)
        self.history_frame.grid(row=0, column=1, rowspan=10, sticky="nsew")
        self.history_frame.grid_columnconfigure(0, weight=1)
        self.history_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(self.history_frame, text="Prediction History", font=("TkDefaultFont", 20, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        # Treeview columns
        tree_columns = ("timestamp", "730nm", "760nm", "810nm", "860nm", "900nm", "940nm", "prediction", "confidence", "leds")
        self.history_tree = ttk.Treeview(
            self.history_frame,
            columns=tree_columns,
            height=15,
            show="tree headings",
            selectmode="extended"
        )

        self.history_tree.bind("<Button-1>", self._on_history_row_click)

        # Define column headings and widths
        self.history_tree.heading("#0", text="")
        self.history_tree.column("#0", width=0, stretch=tk.NO)
        
        column_config = {
            "timestamp": ("Timestamp", 160),
            "730nm": ("730nm", 70),
            "760nm": ("760nm", 70),
            "810nm": ("810nm", 70),
            "860nm": ("860nm", 70),
            "900nm": ("900nm", 70),
            "940nm": ("940nm", 70),
            "prediction": ("Prediction", 100),
            "confidence": ("Confidence", 100),
            "leds": ("LEDs", 80),
        }

        for col, (heading, width) in column_config.items():
            self.history_tree.heading(col, text=heading)
            self.history_tree.column(col, width=width, anchor="center")

        self.history_tree.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(self.history_frame, orient="vertical", command=self.history_tree.yview)
        v_scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 10))
        h_scrollbar = ttk.Scrollbar(self.history_frame, orient="horizontal", command=self.history_tree.xview)
        h_scrollbar.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.history_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Button frame
        button_frame = ttk.Frame(self.history_frame)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 0))
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)

        self.delete_button = ttk.Button(button_frame, text="Delete Selected", command=self._delete_selected_entry)
        self.delete_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.back_button = ttk.Button(button_frame, text="Back to Main", command=self.show_prediction_frame)
        self.back_button.grid(row=0, column=1, sticky="ew", padx=(2, 2))

        self.refresh_button = ttk.Button(button_frame, text="Refresh", command=self._load_history)
        self.refresh_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def show_history_frame(self):
        """Switch to history frame and load data."""
        self.main_frame.grid_remove()
        self.history_frame.grid()
        self._load_history()

    def show_prediction_frame(self):
        """Switch back to prediction frame."""
        self.history_frame.grid_remove()
        self.main_frame.grid()

    def _on_history_row_click(self, event):
        item = self.history_tree.identify_row(event.y)
        if not item:
            return "break"

        if item in self.history_tree.selection():
            self.history_tree.selection_remove(item)
        else:
            self.history_tree.selection_add(item)

        return "break"

    def _load_history(self):
        """Load prediction history from CSV into Treeview."""
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        if not os.path.isfile(HISTORY_PATH):
            return

        try:
            with open(HISTORY_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    values = (
                        row.get("timestamp", ""),
                        f"{float(row.get('ch_730', 0)):.4f}",
                        f"{float(row.get('ch_760', 0)):.4f}",
                        f"{float(row.get('ch_810', 0)):.4f}",
                        f"{float(row.get('ch_860', 0)):.4f}",
                        f"{float(row.get('ch_900', 0)):.4f}",
                        f"{float(row.get('ch_940', 0)):.4f}",
                        row.get("prediction", ""),
                        row.get("confidence", ""),
                        row.get("leds", ""),
                    )
                    self.history_tree.insert("", "end", values=values)
        except Exception as e:
            print(f"Error loading history: {e}")

    def _save_prediction_result(self, predicted_label, confidence_text, averaged_scan, led_flags):
        """Save prediction result to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
            leds_str = f"W={led_flags[0]} IR={led_flags[1]} UV={led_flags[2]}"
            
            # Extract numeric confidence value - handle different formats
            confidence_value = "N/A"
            if "Confidence: " in confidence_text:
                # Extract the percentage value
                conf_part = confidence_text.split("Confidence: ")[1].split("%")[0]
                try:
                    confidence_value = str(float(conf_part.strip()) / 100)  # Convert to decimal
                except ValueError:
                    confidence_value = "N/A"
            
            row = {
                "timestamp": timestamp,
                "ch_730": averaged_scan[0],
                "ch_760": averaged_scan[1],
                "ch_810": averaged_scan[2],
                "ch_860": averaged_scan[3],
                "ch_900": averaged_scan[4],
                "ch_940": averaged_scan[5],
                "prediction": predicted_label,
                "confidence": confidence_value,
                "leds": leds_str,
            }

            # Check if file exists and has content (with header)
            file_has_header = False
            if os.path.isfile(HISTORY_PATH):
                with open(HISTORY_PATH, "r") as f:
                    first_line = f.readline()
                    file_has_header = "timestamp" in first_line

            # Write with header if needed
            with open(HISTORY_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_has_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"Error saving prediction result: {e}")

    def _delete_selected_entry(self):
        """Delete selected entries from history."""
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select at least one entry to delete.")
            return

        count = len(selected)
        confirm_message = f"Delete the selected {count} entr{'y' if count == 1 else 'ies'}?"
        if not messagebox.askyesno("Confirm Delete", confirm_message):
            return

        selected_timestamps = set()
        for item in selected:
            values = self.history_tree.item(item, "values")
            if values:
                selected_timestamps.add(values[0])

        try:
            # Use a temp file for atomic write
            temp_path = HISTORY_PATH + ".tmp"
            rows_kept = 0
            if os.path.isfile(HISTORY_PATH):
                with open(HISTORY_PATH, "r", newline="") as f_in:
                    reader = csv.DictReader(f_in)
                    fieldnames = reader.fieldnames
                    with open(temp_path, "w", newline="") as f_out:
                        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in reader:
                            if row.get("timestamp") not in selected_timestamps:
                                writer.writerow(row)
                                rows_kept += 1

                # Atomic replace - rename temp file to actual file
                if os.path.exists(temp_path):
                    os.replace(temp_path, HISTORY_PATH)

            # Refresh the display
            time.sleep(0.1)  # Small delay to ensure file write completes
            self._load_history()
            messagebox.showinfo("Success", f"Deleted {count} entr{'y' if count == 1 else 'ies'}. {rows_kept} rows remaining.")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            messagebox.showerror("Error", f"Failed to delete entry: {e}")


def main():
    root = tk.Tk()
    PredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()