from neraium_core.models import SignalDefinition
import inspect

print("=== SignalDefinition Fields ===")
print(SignalDefinition.model_fields)
print()

print("=== SignalDefinition Signature ===")
print(inspect.signature(SignalDefinition.__init__))
print()

print("=== SignalDefinition Schema ===")
print(SignalDefinition.model_json_schema())
