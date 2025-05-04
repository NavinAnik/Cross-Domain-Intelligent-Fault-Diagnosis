import time

def cyber_prank():
    fake_warning = """
    [CRITICAL ERROR] GPU Overload Detected!
    ================================
    TensorFlow has detected an anomaly in GPU core temperature (≈ 313°C).
    Possible causes:
    - Training too hard.
    - Black hole formation.
    - Code too optimized.
    
    Attempting automatic cooldown...
    """
    for line in fake_warning.strip().split('\n'):
        print(line)
        time.sleep(0.5)
    for i in range(5, 0, -1):
        print(f"Cooling down in... {i}")
        time.sleep(0.7)
    print("System stabilized. Just kidding 😎 – training complete!\n")