import random
import time

import requests

URL = "http://127.0.0.1:8000/telemetry"

while True:
    payload = {"cpu_usage": random.randint(10, 100), "memory_usage": random.randint(20, 100)}
    try:
        response = requests.post(URL, json=payload, timeout=5)
        print("Sent:", payload, "Response:", response.json())
    except Exception as exc:
        print("Error:", exc)
    time.sleep(1)
