#!/usr/bin/env python3
import os
import sys
import csv
import time
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import qwiic_as7265x


BUTTON_LABELS = [
    "Sweet",
    "Not Sweet"
]

# Columns for the data table
# Display columns (first is row number for UI only)
COLUMN_HEADERS = [
    ("No", "No"),
    ("Class", "Classification"),
    ("Wh", "White"),
    ("IR", "IR"),
    ("UV", "UV"),
    ("730nm", "730nm"),
    ("760nm", "760nm"),
    ("810nm", "810nm"),
    ("860nm", "860nm"),
    ("900nm", "900nm"),
    ("940nm", "940nm"),
    ("Delete", "Delete"),  # pseudo-button column
]

# CSV columns are independent of display (no row number, no Delete)
CSV_HEADERS = ["Class", "Wh", "IR", "UV", "730nm", "760nm", "810nm", "860nm", "900nm", "940nm"]

# Pagination size for UI
PAGE_SIZE = 50
SCAN_COUNT = 32
SCAN_DELAY_SECONDS = 0.1


class CornGUI:
    def __init__(self, root, csv_file="corn_data.csv"):
        self.root = root
        self.csv_file = csv_file
        self.fullscreen = False

        self.root.title("Corn Spectroscopy Data Collector")
        # Maximized on startup (not fullscreen)
        try:
            self.root.state('zoomed')
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                self.root.geometry("800x480")
        self.root.bind("<Escape>", self.toggle_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)

        # Grid: 3 rows for approx 80% (table), 10% (buttons), 10% (spacer)
        # 3 columns for approx 5% (left), 90% (content), 5% (right)
        for c, w in enumerate((1, 15, 1)):
            self.root.grid_columnconfigure(c, weight=w, uniform="cols")
        # Adjust row weight distribution to give more vertical space to buttons/check boxes
        for r, w in enumerate((15, 3, 2)):
            self.root.grid_rowconfigure(r, weight=w, uniform="rows")

        # Data store mirrors table contents for CSV rewriting (define BEFORE building table)
        self.rows = []  # list of tuples: (classification, white, ir, uv, 730, 760, 810, 860, 900, 940)
        self.page_size = PAGE_SIZE
        self.current_page = 1  # 1-based

        # Table frame in row 0, col 1
        self.table_frame = ttk.Frame(self.root)
        self.table_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self._build_table_container()

        # Buttons frame in row 1, col 1
        self.buttons_frame = ttk.Frame(self.root)
        self.buttons_frame.grid(row=1, column=1, sticky="nsew", padx=6, pady=(0, 6))
        self._build_buttons()
        # Ensure buttons frame rows don't compress checkboxes
        self.buttons_frame.grid_rowconfigure(0, weight=0)
        self.buttons_frame.grid_rowconfigure(1, weight=0)

        # LED selection checkboxes (which bulbs to illuminate)
        self.led_frame = ttk.Frame(self.buttons_frame)
        self.led_frame.grid(row=1, column=0, columnspan=len(BUTTON_LABELS), sticky="w", padx=6)
        self.led_white_var = tk.BooleanVar(value=True)
        self.led_ir_var = tk.BooleanVar(value=True)
        self.led_uv_var = tk.BooleanVar(value=True)
        self.led_checkbuttons = [
            ttk.Checkbutton(self.led_frame, text="White LED", variable=self.led_white_var),
            ttk.Checkbutton(self.led_frame, text="IR LED", variable=self.led_ir_var),
            ttk.Checkbutton(self.led_frame, text="UV LED", variable=self.led_uv_var),
        ]
        self.led_checkbuttons[0].grid(row=0, column=0, padx=(0,12))
        self.led_checkbuttons[1].grid(row=0, column=1, padx=(0,12))
        self.led_checkbuttons[2].grid(row=0, column=2, padx=(0,12))

        self.capture_progress = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.buttons_frame, orient="horizontal", mode="determinate",
                                            maximum=SCAN_COUNT, variable=self.capture_progress)
        self.progress_bar.grid(row=2, column=0, columnspan=len(BUTTON_LABELS), sticky="ew", padx=6, pady=(0, 6))
        self.capture_in_progress = False

        # Status bar in bottom spacer area
        self.status_var = tk.StringVar(value="Initializing sensor...")
        self.status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        self.status.grid(row=2, column=1, sticky="ew", padx=6, pady=(0, 6))

        # CSV header ensure
        self._ensure_csv_header()
        # Load any existing rows from CSV and show them
        self._load_existing_csv()
        self._rebuild_table(force_last=True)

        # Sensor init
        self.sensor = None
        self._init_sensor()

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        try:
            self.root.attributes("-fullscreen", self.fullscreen)
        except tk.TclError:
            pass
        if not self.fullscreen:
            self.root.geometry("800x480")

    # ------------------------- UI: Table -------------------------
    def _build_table_container(self):
        # Grid config for table frame
        self.table_frame.grid_columnconfigure(0, weight=1)
        # Row 0: scrollable table, Row 1: h-scroll, Row 2: pager
        self.table_frame.grid_rowconfigure(0, weight=1)
        self.table_frame.grid_rowconfigure(1, weight=0)
        self.table_frame.grid_rowconfigure(2, weight=0)

        # Canvas for scrollable area
        self.table_canvas = tk.Canvas(self.table_frame, highlightthickness=0)
        self.table_canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        self.v_scroll = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.table_canvas.yview)
        self.h_scroll = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.table_canvas.xview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.table_canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        # Inner frame that holds grid cells
        self.table_inner = ttk.Frame(self.table_canvas)
        self.table_window = self.table_canvas.create_window((0, 0), window=self.table_inner, anchor="nw")

        # Bind resize to update scrollregion
        self.table_inner.bind("<Configure>", lambda e: self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all")))
        self.table_canvas.bind("<Configure>", self._on_canvas_resize)

        # Define header style (bold + border appearance)
        style = ttk.Style(self.root)
        # Use a default font derivation; on some systems 'TkDefaultFont' exists
        try:
            default_font = style.lookup("TLabel", "font") or "TkDefaultFont"
        except Exception:
            default_font = "TkDefaultFont"
        style.configure("Header.TLabel", font=(default_font, 10, "bold"), padding=(6, 4))

        # Build initial paginator
        self._build_paginator()
        self._rebuild_table(force_last=True)

    def _total_pages(self):
        return max(1, (len(self.rows) + self.page_size - 1) // self.page_size)

    def _build_paginator(self):
        self.pager_frame = ttk.Frame(self.table_frame)
        self.pager_frame.grid(row=2, column=0, sticky="ew", padx=2, pady=(4, 0))
        for i in range(5):
            self.pager_frame.grid_columnconfigure(i, weight=1 if i == 2 else 0)
        self.first_btn = ttk.Button(self.pager_frame, text="<< First", command=self._go_first)
        self.prev_btn = ttk.Button(self.pager_frame, text="< Prev", command=self._go_prev)
        self.next_btn = ttk.Button(self.pager_frame, text="Next >", command=self._go_next)
        self.last_btn = ttk.Button(self.pager_frame, text="Last >>", command=self._go_last)
        self.page_label = ttk.Label(self.pager_frame, text="Page 1/1 (0)", anchor="center")
        self.first_btn.grid(row=0, column=0, padx=2)
        self.prev_btn.grid(row=0, column=1, padx=2)
        self.page_label.grid(row=0, column=2, padx=2, sticky="ew")
        self.next_btn.grid(row=0, column=3, padx=2)
        self.last_btn.grid(row=0, column=4, padx=2)

    def _update_paginator(self):
        total_pages = self._total_pages()
        total_rows = len(self.rows)
        self.page_label.configure(text=f"Page {self.current_page}/{total_pages} ({total_rows})")
        # Enable/disable buttons
        self.first_btn.configure(state=("disabled" if self.current_page <= 1 else "normal"))
        self.prev_btn.configure(state=("disabled" if self.current_page <= 1 else "normal"))
        self.next_btn.configure(state=("disabled" if self.current_page >= total_pages else "normal"))
        self.last_btn.configure(state=("disabled" if self.current_page >= total_pages else "normal"))

    def _go_first(self):
        self.current_page = 1
        # After deleting, keep current page clamped and refresh
        self._rebuild_table()

    def _go_prev(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._rebuild_table()

    def _go_next(self):
        if self.current_page < self._total_pages():
            self.current_page += 1
            self._rebuild_table()

    def _go_last(self):
        self.current_page = self._total_pages()
        self._rebuild_table(force_last=True)

    def _on_canvas_resize(self, event):
        # Expand inner frame width to canvas width (for horizontal scrollbar behaviour)
        self.table_canvas.itemconfigure(self.table_window, width=event.width)

    def _rebuild_table(self, force_last=False):
        # Clear existing widgets
        for child in self.table_inner.winfo_children():
            child.destroy()

        # Configure columns
        for col in range(len(COLUMN_HEADERS)):
            if col in (2, 3, 4):
                self.table_inner.grid_columnconfigure(col, weight=0)
            else:
                self.table_inner.grid_columnconfigure(col, weight=1)

        # Render headers in row 0
        headers = [h for h, _ in COLUMN_HEADERS]
        for col, header in enumerate(headers):
            if col in (2, 3, 4):
                lbl = ttk.Label(self.table_inner, text=header, anchor="center", style="Header.TLabel", width=3)
            else:
                lbl = ttk.Label(self.table_inner, text=header, anchor="center", style="Header.TLabel")
            lbl.grid(row=0, column=col, sticky="nsew", padx=1, pady=(0, 4))
            lbl.configure(relief="ridge")

        # Pagination setup (oldest -> newest display)
        total_pages = self._total_pages()
        if force_last:
            self.current_page = total_pages
        else:
            self.current_page = max(1, min(self.current_page, total_pages))
        start = (self.current_page - 1) * self.page_size
        end = min(len(self.rows), start + self.page_size)
        rows_to_display = self.rows[start:end]

        for r_index, row in enumerate(rows_to_display, start=1):
            classification, white, ir, uv, t, u, v, wv, k, l = row
            row_no = start + r_index  # 1-based overall row number
            display_values = [row_no, classification, white, ir, uv, t, u, v, wv, k, l]
            for c_index, value in enumerate(display_values):
                # Fix width for LED logic columns
                if c_index in (2, 3, 4):
                    lbl = ttk.Label(self.table_inner, text=f"{value}" if not isinstance(value, (float, int)) else f"{int(value)}", anchor="center", padding=(4, 2), width=3)
                else:
                    lbl = ttk.Label(self.table_inner, text=f"{value:.6g}" if isinstance(value, (float, int)) else str(value), anchor="center", padding=(4, 2))
                lbl.grid(row=r_index, column=c_index, sticky="nsew", padx=1, pady=4)
            # Delete button (only this cell is red)
            # Map displayed r_index to actual index in full list
            real_index = start + r_index - 1
            del_btn = tk.Button(self.table_inner, text="Delete", bg="#5a0000", fg="white", relief="raised",
                                command=lambda idx=real_index: self._delete_row(idx))
            del_btn.grid(row=r_index, column=len(display_values), sticky="nsew", padx=1, pady=4)

        # Update paginator and scroll to bottom
        self._update_paginator()
        self.table_canvas.update_idletasks()
        # Force scrollregion update before scrolling
        self.table_canvas.configure(scrollregion=self.table_canvas.bbox("all"))
        self.table_canvas.yview_moveto(1.0)

    def _delete_row(self, index):
        if 0 <= index < len(self.rows):
            del self.rows[index]
            self._rewrite_csv()
            self._rebuild_table()
            self.status_var.set("Row deleted.")

    # (Treeview click handler removed; custom grid used instead)

    # ------------------------- UI: Buttons -------------------------
    def _build_buttons(self):
        # Put buttons centered and spread horizontally
        for i in range(len(BUTTON_LABELS)):
            self.buttons_frame.grid_columnconfigure(i, weight=1)

        for idx, label in enumerate(BUTTON_LABELS):
            btn = ttk.Button(self.buttons_frame, text=label,
                             command=lambda l=label: self.capture_sample(l))
            btn.grid(row=0, column=idx, padx=6, pady=6, sticky="nsew")

        # Disable during no-sensor
        self.capture_buttons = [w for w in self.buttons_frame.winfo_children() if isinstance(w, ttk.Button)]

    # ------------------------- Sensor -------------------------
    def _init_sensor(self):
        try:
            s = qwiic_as7265x.QwiicAS7265x()
            if not s.is_connected():
                self.status_var.set("Sensor not connected. Check wiring and I2C.")
                self._set_buttons_state("disabled")
                return
            if not s.begin():
                self.status_var.set("Failed to initialize sensor.")
                self._set_buttons_state("disabled")
                return
            # Configure recommended acquisition params
            s.set_integration_cycles(50)  # ~140ms, doubled for mode 3
            s.set_gain(s.kGain64x)
            s.disable_indicator()
            self.sensor = s
            self.status_var.set("Sensor ready.")
            self._set_buttons_state("normal")
        except Exception as e:
            self.status_var.set(f"Sensor error: {e}")
            self._set_buttons_state("disabled")

    def _set_buttons_state(self, state):
        for b in getattr(self, "capture_buttons", []):
            try:
                b.configure(state=state)
            except Exception:
                pass
        for checkbutton in getattr(self, "led_checkbuttons", []):
            try:
                checkbutton.configure(state=state)
            except Exception:
                pass

    # ------------------------- CSV Helpers -------------------------
    def _backup_path(self):
        return f"{self.csv_file}.bak"

    def _ensure_csv_header(self):
        needs_header = not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0
        if needs_header:
            try:
                tmp = f"{self.csv_file}.tmp"
                with open(tmp, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(CSV_HEADERS)
                os.replace(tmp, self.csv_file)
                # Create initial backup of empty file with header
                try:
                    shutil.copyfile(self.csv_file, self._backup_path())
                except Exception:
                    pass
            except OSError as e:
                messagebox.showerror("CSV Error", f"Unable to write CSV header: {e}")

    def _atomic_write_rows(self, rows):
        tmp = f"{self.csv_file}.tmp"
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADERS)
                for r in rows:
                    w.writerow(r)
            os.replace(tmp, self.csv_file)
            try:
                shutil.copyfile(self.csv_file, self._backup_path())
            except Exception:
                pass
        except Exception as e:
            # Clean tmp if present
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            raise e

    def _rewrite_csv(self):
        try:
            self._atomic_write_rows(self.rows)
        except OSError as e:
            messagebox.showerror("CSV Error", f"Unable to update CSV: {e}\nAttempting to restore backup...")
            self._restore_backup_and_reload()

    def _append_csv(self, row):
        # Append-only for CSV (oldest-first). Then refresh backup copy.
        try:
            # If file does not exist yet, ensure header first
            if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
                self._ensure_csv_header()
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(row)
            try:
                shutil.copyfile(self.csv_file, self._backup_path())
            except Exception:
                pass
        except OSError as e:
            messagebox.showerror("CSV Error", f"Unable to append CSV: {e}\nAttempting to restore backup...")
            self._restore_backup_and_reload()

    def _load_existing_csv(self):
        if not os.path.exists(self.csv_file):
            # Try restoring from backup if main is missing
            bak = self._backup_path()
            if os.path.exists(bak):
                try:
                    shutil.copyfile(bak, self.csv_file)
                except Exception:
                    return
            else:
                return
        try:
            with open(self.csv_file, "r", newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                header = next(r, None)
                # Basic header validation; tolerate extra columns
                if header is None:
                    raise OSError("Missing header")
                # Build index map per header, allowing missing columns (defaults)
                idx_map = []
                def index_of_any(names, header_list):
                    for name in names:
                        if name in header_list:
                            return header_list.index(name)
                    return None

                # Support legacy headers too
                idx_map = []
                idx_map.append(index_of_any(["Class", "Classification"], header))  # classification
                idx_map.append(index_of_any(["Wh", "White"], header))             # white flag
                idx_map.append(index_of_any(["IR"], header))                        # ir flag
                idx_map.append(index_of_any(["UV"], header))                        # uv flag
                idx_map.append(index_of_any(["730nm"], header))
                idx_map.append(index_of_any(["760nm"], header))
                idx_map.append(index_of_any(["810nm"], header))
                idx_map.append(index_of_any(["860nm"], header))
                idx_map.append(index_of_any(["900nm"], header))
                idx_map.append(index_of_any(["940nm"], header))

                loaded = []
                for row in r:
                    try:
                        def get_val(idx, cast=None, default=None):
                            if idx is None:
                                return default
                            if idx >= len(row):
                                return default
                            val = row[idx]
                            if cast:
                                return cast(val)
                            return val

                        classification = get_val(idx_map[0], cast=None, default="")
                        # LED flags default to 0 if missing
                        white = int(float(get_val(idx_map[1], cast=float, default=0))) if idx_map[1] is not None else 0
                        ir = int(float(get_val(idx_map[2], cast=float, default=0))) if idx_map[2] is not None else 0
                        uv = int(float(get_val(idx_map[3], cast=float, default=0))) if idx_map[3] is not None else 0
                        t = float(get_val(idx_map[4], cast=float, default=0.0))
                        u = float(get_val(idx_map[5], cast=float, default=0.0))
                        v = float(get_val(idx_map[6], cast=float, default=0.0))
                        wv = float(get_val(idx_map[7], cast=float, default=0.0))
                        k = float(get_val(idx_map[8], cast=float, default=0.0))
                        l = float(get_val(idx_map[9], cast=float, default=0.0))
                        loaded.append((classification, white, ir, uv, t, u, v, wv, k, l))
                    except Exception:
                        # Skip malformed rows
                        continue
                # Keep chronological order in memory (oldest->newest) and do not trim CSV
                self.rows = loaded
        except OSError as e:
            messagebox.showerror("CSV Error", f"Unable to read CSV: {e}\nAttempting to restore backup...")
            self._restore_backup_and_reload()

    def _restore_backup_and_reload(self):
        bak = self._backup_path()
        if os.path.exists(bak):
            try:
                shutil.copyfile(bak, self.csv_file)
                self._load_existing_csv()
                self._rebuild_table()
                self.status_var.set("Restored from backup.")
            except Exception as e:
                messagebox.showerror("Backup Error", f"Failed to restore backup: {e}")
        else:
            messagebox.showwarning("Backup Missing", "No backup CSV available to restore.")

    def _apply_snv(self, values):
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_dev = variance ** 0.5
        if std_dev == 0:
            return [0.0 for _ in values]
        return [(value - mean_value) / std_dev for value in values]

    def _update_capture_progress(self, completed_scans):
        self.capture_progress.set(completed_scans)
        self.status_var.set(f"Capturing scan {completed_scans}/{SCAN_COUNT}...")

    def _finish_capture_success(self, row):
        self.rows.append(row)
        self._append_csv(row)
        self._rebuild_table(force_last=True)
        self.capture_progress.set(0)
        self.capture_in_progress = False
        self.status_var.set(f"Sample captured from {SCAN_COUNT} averaged SNV scans.")
        self._set_buttons_state("normal")

    def _finish_capture_error(self, error_message):
        self.capture_progress.set(0)
        self.capture_in_progress = False
        self._set_buttons_state("normal")
        messagebox.showerror("Read Error", f"Failed to read sensor: {error_message}")
        self.status_var.set("Capture failed.")

    def _capture_sample_worker(self, classification, enabled_devices, white_flag, ir_flag, uv_flag):
        try:
            snv_scans = []
            for dev in enabled_devices:
                self.sensor.enable_bulb(dev)

            try:
                for scan_index in range(SCAN_COUNT):
                    self.sensor.take_measurements()
                    raw_scan = [
                        self.sensor.get_calibrated_t(),
                        self.sensor.get_calibrated_u(),
                        self.sensor.get_calibrated_v(),
                        self.sensor.get_calibrated_w(),
                        self.sensor.get_calibrated_k(),
                        self.sensor.get_calibrated_l(),
                    ]
                    snv_scans.append(self._apply_snv(raw_scan))

                    self.root.after(0, self._update_capture_progress, scan_index + 1)
                    if scan_index < SCAN_COUNT - 1:
                        time.sleep(SCAN_DELAY_SECONDS)
            finally:
                for dev in enabled_devices:
                    self.sensor.disable_bulb(dev)

            averaged_scan = [sum(scan[channel] for scan in snv_scans) / len(snv_scans) for channel in range(len(snv_scans[0]))]
            t, u, v, wv, k, l = averaged_scan
            row = (classification, white_flag, ir_flag, uv_flag, t, u, v, wv, k, l)
            self.root.after(0, self._finish_capture_success, row)
        except Exception as e:
            self.root.after(0, self._finish_capture_error, str(e))

    # ------------------------- Data Ops -------------------------
    def capture_sample(self, classification):
        if self.sensor is None:
            messagebox.showwarning("No Sensor", "Sensor is not ready.")
            return
        if self.capture_in_progress:
            messagebox.showinfo("Capture Running", "Please wait for the current capture to finish.")
            return
        enabled_devices = []
        try:
            if self.led_white_var.get():
                enabled_devices.append(self.sensor.kLedWhite)
            if self.led_ir_var.get():
                enabled_devices.append(self.sensor.kLedIr)
            if self.led_uv_var.get():
                enabled_devices.append(self.sensor.kLedUv)
            white_flag = 1 if self.led_white_var.get() else 0
            ir_flag = 1 if self.led_ir_var.get() else 0
            uv_flag = 1 if self.led_uv_var.get() else 0
        except Exception as e:
            messagebox.showerror("Read Error", f"Failed to read sensor: {e}")
            return
        self.capture_in_progress = True
        self.capture_progress.set(0)
        self.status_var.set(f"Capturing scan 0/{SCAN_COUNT}...")
        self._set_buttons_state("disabled")
        threading.Thread(
            target=self._capture_sample_worker,
            args=(classification, enabled_devices, white_flag, ir_flag, uv_flag),
            daemon=True,
        ).start()

    # Legacy Treeview insertion method removed; rebuild happens after data append.

    # Old Treeview deletion method removed (handled by _delete_row index now)


def main():
    root = tk.Tk()
    app = CornGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
