from trl import SFTConfig, SFTTrainer
import inspect

print("SFTConfig args:")
print(inspect.signature(SFTConfig.__init__))

print("\nSFTTrainer args:")
print(inspect.signature(SFTTrainer.__init__))
