from typing import List

class Signal:
    def __init__(self, id: int, value: int):
        self.id = id
        self.value = value

class MergedSignal:
    def __init__(self):
        self.signals = []
    
    def combine(self, signal: Signal):
        self.signals.append(signal)
    
    def __str__(self):
        return f"MergedSignal({self.signals})"

class SignalProcessor:
    def __init__(self):
        self.signals = []
    
    def add_signal(self, signal: Signal):
        self.signals.append(signal)
    
    def merge_signals(self, signals: List[Signal]) -> MergedSignal:
        # Original: Nested loops leading to O(n^2)
        # Optimized: Use a dictionary for O(n) lookup
        signal_map = {sig.id: sig for sig in signals}
        merged = MergedSignal()
        for sig_id in signal_map:
            merged.combine(signal_map[sig_id])
        return merged

# Example usage
processor = SignalProcessor()
processor.add_signal(Signal(1, 5))
processor.add_signal(Signal(2, 3))
processor.add_signal(Signal(3, 7))

merged_signal = processor.merge_signals([Signal(1, 5), Signal(2, 3), Signal(3, 7)])
print(merged_signal)