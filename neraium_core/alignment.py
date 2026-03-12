class AlignmentEngine:
    def __init__(self, system_definition):
        self.system_definition = system_definition

    def align(self, signals: dict):
        """
        Convert signal dictionary into ordered vector based on system definition.
        """

        vector = []

        for signal_name in self.system_definition.vector_order:
            value = signals.get(signal_name, 0.0)
            vector.append(float(value))

        return vector
