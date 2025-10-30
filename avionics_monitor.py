import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random, time, csv, os
from collections import deque
import threading
import datetime
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

SAMPLE_INTERVAL_MS = 500            
PLOT_WINDOW_SEC = 30                
LOG_CSV = "fault_log.csv"          

SENSORS = [
    {"id": "S1", "name": "Altitude Sensor", "nominal": 10000.0, "tol": 200.0},
    {"id": "S2", "name": "Airspeed Sensor",  "nominal": 250.0,   "tol": 15.0},
    {"id": "S3", "name": "Gyro (pitch)",      "nominal": 0.0,     "tol": 2.0},
]

ERROR_CODES = {
    "E01": {"desc": "Sensor Out-of-Range", "severity": "HIGH", "recommend": "Replace sensor or verify wiring."},
    "E02": {"desc": "Intermittent Signal", "severity": "MEDIUM", "recommend": "Inspect connectors; run continuity test."},
    "E03": {"desc": "Stuck at Value", "severity": "HIGH", "recommend": "Replace sensor; check power rails."},
    "E04": {"desc": "Drift Detected", "severity": "LOW", "recommend": "Calibrate sensor; monitor for progression."},
    "E05": {"desc": "Noisy Signal", "severity": "LOW", "recommend": "Check shielding; apply filtering."},
    "OK":  {"desc": "All OK", "severity": "NONE", "recommend": "No action required."}
}

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_log_csv(row):
    header = ["timestamp", "sensor_id", "sensor_name", "code", "description", "severity", "value", "details"]
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

class SensorSimulator:
    def __init__(self, config):
        self.id = config["id"]
        self.name = config["name"]
        self.nominal = float(config["nominal"])
        self.tol = float(config["tol"])
        self.time = 0.0
        self.value = self.nominal
        self.noise_level = 0.005 * abs(self.nominal) + 0.1
        # internal bug flags
        self.forced_fault = None   # set by GUI to inject fault
        self.drift_rate = 0.0
        self.stuck_value = None
        self.last_values = deque(maxlen=10)

    def step(self, dt):
        self.time += dt
        # If stuck fault
        if self.stuck_value is not None:
            self.value = self.stuck_value
            self.last_values.append(self.value)
            return self.value

        self.value = self.nominal + self.drift_rate * self.time

        noise = random.gauss(0, self.noise_level)
        self.value += noise

        if self.forced_fault == "spike":
            # occasional spikes
            if random.random() < 0.02:
                self.value += random.choice([5, -5]) * self.tol
        elif self.forced_fault == "noisy":
            self.value += random.gauss(0, 3 * self.noise_level)
        elif self.forced_fault == "drift":
            self.drift_rate = 0.1 * self.tol
        elif self.forced_fault == "stuck":
            self.stuck_value = self.nominal
        elif self.forced_fault == "out_of_range":
            self.value = self.nominal + 10 * self.tol

        self.last_values.append(self.value)
        return self.value

    def reset_faults(self):
        self.forced_fault = None
        self.drift_rate = 0.0
        self.stuck_value = None

# ---------- Fault detection & classification ----------
def bite_check(sensor: SensorSimulator, value):
    # Basic checks: out-of-range, stuck, noisy, drifting, intermittent
    nominal = sensor.nominal
    tol = sensor.tol
    recent = list(sensor.last_values)
    # If empty recent, small ok
    if not recent:
        recent = [value]

    # Out of range
    if abs(value - nominal) > 5 * tol:
        return "E01", f"Value {value:.2f} outside safe range (nom {nominal})"

    # Stuck: low variance and close to nominal but not changing
    if len(recent) >= 6 and np.std(recent) < 1e-6:
        return "E03", "No variation in readings — possible stuck sensor"

    # Noisy: high std dev relative to tolerance
    if np.std(recent) > 0.5 * tol:
        return "E05", f"High noise (std={np.std(recent):.2f})"

    # Drift: mean deviates slowly over time
    if abs(np.mean(recent) - nominal) > 1.5 * tol:
        return "E04", f"Mean shifted by {np.mean(recent)-nominal:.2f} from nominal"

    # Intermittent: occasional large deltas
    if any(abs(recent[i] - recent[i-1]) > 2 * tol for i in range(1, len(recent))):
        return "E02", "Intermittent large deltas observed"

    return "OK", "Passed BITE"

# ---------- GUI Application ----------
class AvionicsMonitorApp:
    def __init__(self, root):
        self.root = root
        root.title("Avionics Fault Diagnosis & Monitoring Simulator")
        root.geometry("1100x700")

        # Setup sensors
        self.sensors = [SensorSimulator(cfg) for cfg in SENSORS]
        self.start_time = time.time()
        self.data_buffers = {s.id: deque() for s in self.sensors}  # (timestamp, value)

        # Create UI frames
        self.create_controls_frame()
        self.create_plot_frame()
        self.create_status_frame()
        self.create_log_frame()

        # Ensure log file exists
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w") as f:
                f.write("")

        # Start periodic sampling
        self.running = True
        self.last_update = time.time()
        self.root.after(SAMPLE_INTERVAL_MS, self.update_loop)

    def create_controls_frame(self):
        frm = ttk.LabelFrame(self.root, text="Controls / Fault Injection", padding=8)
        frm.place(x=10, y=10, width=350, height=200)

        ttk.Label(frm, text="Select Sensor:").grid(row=0, column=0, sticky="w")
        self.sensor_combo = ttk.Combobox(frm, values=[f"{s.id} - {s.name}" for s in self.sensors], state="readonly")
        self.sensor_combo.current(0)
        self.sensor_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frm, text="Inject Fault:").grid(row=1, column=0, sticky="w")
        self.fault_choice = ttk.Combobox(frm, values=["none", "spike", "noisy", "drift", "stuck", "out_of_range"], state="readonly")
        self.fault_choice.current(0)
        self.fault_choice.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(frm, text="Apply Fault", command=self.apply_fault).grid(row=2, column=0, pady=8)
        ttk.Button(frm, text="Clear Faults (All)", command=self.clear_faults).grid(row=2, column=1, pady=8)

        # BITE run
        ttk.Button(frm, text="Run BITE (All Sensors)", command=self.run_bite_all).grid(row=3, column=0, columnspan=2, pady=8)
        ttk.Label(frm, text="Sample Interval (ms):").grid(row=4, column=0, sticky="w")
        self.interval_var = tk.IntVar(value=SAMPLE_INTERVAL_MS)
        tk.Entry(frm, textvariable=self.interval_var, width=10).grid(row=4, column=1, sticky="w")

    def create_plot_frame(self):
        frm = ttk.LabelFrame(self.root, text="Real-time Sensor Trends", padding=6)
        frm.place(x=370, y=10, width=720, height=420)

        # Matplotlib figure
        self.fig = Figure(figsize=(7,4), dpi=100)
        self.axes = {}
        for i, s in enumerate(self.sensors):
            ax = self.fig.add_subplot(len(self.sensors), 1, i+1)
            ax.set_ylabel(f"{s.id} ({s.name.split()[0]})")
            ax.grid(True)
            self.axes[s.id] = ax
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=frm)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_status_frame(self):
        frm = ttk.LabelFrame(self.root, text="System Health Status", padding=6)
        frm.place(x=10, y=220, width=350, height=210)

        self.health_labels = {}
        for i, s in enumerate(self.sensors):
            ttk.Label(frm, text=f"{s.id} - {s.name}").grid(row=i, column=0, sticky="w", padx=4)
            lbl = ttk.Label(frm, text="UNKNOWN", background="lightgrey", width=30)
            lbl.grid(row=i, column=1, padx=4, pady=4)
            self.health_labels[s.id] = lbl

        # Maintenance recommendation area
        ttk.Label(frm, text="Maintenance Recommendation:").grid(row=4, column=0, sticky="w", pady=(8,0))
        self.reco_text = tk.StringVar(value="No recommendations yet.")
        ttk.Label(frm, textvariable=self.reco_text, wraplength=260).grid(row=5, column=0, columnspan=2, pady=4)

    def create_log_frame(self):
        frm = ttk.LabelFrame(self.root, text="Fault Log & BITE Output", padding=6)
        frm.place(x=10, y=440, width=1080, height=250)
        self.log_widget = scrolledtext.ScrolledText(frm, height=10)
        self.log_widget.pack(fill="both", expand=True)

        ttk.Button(frm, text="Export Log CSV", command=self.export_csv).pack(side="right", padx=8, pady=6)

    # ---------- Control callbacks ----------
    def apply_fault(self):
        sel = self.sensor_combo.get()
        if not sel:
            return
        sensor_id = sel.split()[0]
        fault_type = self.fault_choice.get()
        sensor = next((s for s in self.sensors if s.id == sensor_id), None)
        if not sensor:
            return
        if fault_type == "none":
            sensor.reset_faults()
            self.log(f"[{timestamp()}] Cleared faults on {sensor.id}")
        else:
            sensor.forced_fault = fault_type
            self.log(f"[{timestamp()}] Injected fault '{fault_type}' into {sensor.id}")

    def clear_faults(self):
        for s in self.sensors:
            s.reset_faults()
        self.log(f"[{timestamp()}] Cleared all injected faults")

    def run_bite_all(self):
        # Run BITE check for all sensors using latest values
        for s in self.sensors:
            buff = list(self.data_buffers[s.id])
            if not buff:
                self.log(f"[{timestamp()}] BITE: {s.id} - No data available")
                continue
            _, latest = buff[-1]
            code, desc = bite_check(s, latest)
            info = ERROR_CODES.get(code, {"desc":"Unknown","severity":"UNKNOWN","recommend":""})
            self.log_bite(s, code, desc, latest)

    def export_csv(self):
        if not os.path.exists(LOG_CSV):
            messagebox.showinfo("Export", "No log CSV present yet.")
        else:
            messagebox.showinfo("Export", f"Log exported as {os.path.abspath(LOG_CSV)}")

    # ---------- Logging ----------
    def log(self, text):
        self.log_widget.insert(tk.END, text + "\n")
        self.log_widget.see(tk.END)

    def log_bite(self, sensor, code, details, value):
        info = ERROR_CODES.get(code, {"desc":"Unknown","severity":"UNKNOWN","recommend":""})
        row = [timestamp(), sensor.id, sensor.name, code, info["desc"], info["severity"], f"{value:.3f}", details]
        append_log_csv(row)
        self.log(f"[BITE][{timestamp()}] {sensor.id} {sensor.name} -> {code}: {info['desc']} | Value={value:.2f} | Severity={info['severity']}")
        # Show maintenance recommendation for high severity
        if info["severity"] in ("HIGH", "MEDIUM"):
            self.reco_text.set(info["recommend"])
        # update color-coded status
        self.update_health_label(sensor.id, info["severity"])

    def update_health_label(self, sensor_id, severity):
        lbl = self.health_labels.get(sensor_id)
        if not lbl:
            return
        if severity == "HIGH":
            lbl.config(text="CRITICAL", background="red")
        elif severity == "MEDIUM":
            lbl.config(text="DEGRADED", background="orange")
        elif severity == "LOW":
            lbl.config(text="ATTENTION", background="yellow")
        else:
            lbl.config(text="OK", background="lightgreen")

    # ---------- Main update loop ----------
    def update_loop(self):
        # adapt interval if user changed
        try:
            interval = int(self.interval_var.get())
        except Exception:
            interval = SAMPLE_INTERVAL_MS
        self.root.after(interval, self.update_loop)

        now = time.time()
        dt = now - self.last_update if self.last_update else 0.0
        self.last_update = now

        # Sample sensors
        for s in self.sensors:
            val = s.step(dt)
            t = now - self.start_time
            self.data_buffers[s.id].append((t, val))

            # keep only PLOT_WINDOW_SEC
            while self.data_buffers[s.id] and (t - self.data_buffers[s.id][0][0]) > PLOT_WINDOW_SEC:
                self.data_buffers[s.id].popleft()

            # Run BITE check each sample (or could be periodic)
            code, details = bite_check(s, val)
            if code != "OK":
                self.log_bite(s, code, details, val)

        # Update plots
        self.refresh_plots()

    def refresh_plots(self):
        for s in self.sensors:
            ax = self.axes[s.id]
            ax.clear()
            data = list(self.data_buffers[s.id])
            if data:
                xs, ys = zip(*data)
                ax.plot(xs, ys)
                ax.set_xlim(max(0, xs[-1]-PLOT_WINDOW_SEC), xs[-1] if xs else PLOT_WINDOW_SEC)
                ax.set_ylabel(f"{s.id}: {s.name.split()[0]}")
                # show nominal and tolerance band
                ax.axhline(s.nominal, linestyle="--", linewidth=0.7)
                ax.fill_between(xs, [s.nominal - s.tol]*len(xs), [s.nominal + s.tol]*len(xs), alpha=0.1)
            else:
                ax.set_ylabel(f"{s.id}")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()

# ---------- Run application ----------
def main():
    root = tk.Tk()
    app = AvionicsMonitorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
