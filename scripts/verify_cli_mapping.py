
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

from src.config.loader import load_yaml, apply_cli_overrides
from src.config.schema import SimulationConfig

def test_cli_mapping():
    print("Testing CLI Flag Mapping...")
    
    # Base config (default enable=True)
    base_config = SimulationConfig()
    # Ensure default is True
    base_config.aft.enable = True
    print(f"Base Config AFT Enable: {base_config.aft.enable}")
    
    # Test 1: --no-aft
    print("\nTest 1: Mocking --no-aft")
    args_no = argparse.Namespace(no_aft=True, use_aft=False)
    config_no = apply_cli_overrides(base_config, args_no)
    print(f"Result AFT Enable: {config_no.aft.enable}")
    
    if config_no.aft.enable is False:
        print("SUCCESS: --no-aft disabled AFT.")
    else:
        print("FAILURE: --no-aft did not disable AFT.")

    # Test 2: --use-aft
    print("\nTest 2: Mocking --use-aft on disabled config")
    base_disabled = SimulationConfig()
    base_disabled.aft.enable = False
    args_yes = argparse.Namespace(no_aft=False, use_aft=True)
    config_yes = apply_cli_overrides(base_disabled, args_yes)
    print(f"Result AFT Enable: {config_yes.aft.enable}")
    
    if config_yes.aft.enable is True:
        print("SUCCESS: --use-aft enabled AFT.")
    else:
        print("FAILURE: --use-aft did not enable AFT.")
        
    # Test 3: No flags
    print("\nTest 3: No flags")
    args_none = argparse.Namespace(no_aft=False, use_aft=False)
    config_none = apply_cli_overrides(base_config, args_none)
    print(f"Result AFT Enable: {config_none.aft.enable}")
    
    if config_none.aft.enable == base_config.aft.enable:
        print("SUCCESS: No flags preserved config.")
    else:
        print("FAILURE: No flags changed config!")

if __name__ == "__main__":
    test_cli_mapping()
