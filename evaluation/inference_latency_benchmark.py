"""
Standalone inference latency test.
Run directly on the deployed Raspberry Pi 4.
"""

import time
import joblib
import numpy as np

MODEL_PATH = "final_model.joblib"

# taken from prediction_history.csv.
sample_input = np.array([[
    -0.3344241643453868,
    -0.42397443299358006,
    -0.21205674255821927,
     2.0254893378712935,
    -0.48322455254864294,
    -0.5737760373747163
]], dtype=float)

N_WARMUP = 10
N_TRIALS = 100

model = joblib.load(MODEL_PATH)

# Warm-up
for _ in range(N_WARMUP):
    model.predict(sample_input)
    model.predict_proba(sample_input)

# Timed runs
latencies_ms = []

for _ in range(N_TRIALS):
    t0 = time.perf_counter()

    model.predict(sample_input)
    model.predict_proba(sample_input)

    t1 = time.perf_counter()

    latencies_ms.append((t1 - t0) * 1000)

latencies_ms = np.array(latencies_ms)

print(f"Warm-up runs: {N_WARMUP}")
print(f"Timed trials: {N_TRIALS}")
print(f"Mean latency:  {latencies_ms.mean():.3f} ms")
print(f"Min latency:   {latencies_ms.min():.3f} ms")
print(f"Max latency:   {latencies_ms.max():.3f} ms")
print(f"Std deviation: {latencies_ms.std():.3f} ms")

print("\nIndividual trial latencies (ms):")
print(latencies_ms)
