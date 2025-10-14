#!/usr/bin/env python3
"""
Patch xformers to disable flash attention 3
This script modifies the xformers dispatch.py file to set _USE_FLASH_ATTENTION_3 = False
"""

import site
import os
import glob
import sys


def patch_xformers():
    """Find and patch the xformers dispatch.py file to disable flash attention 3"""

    # Find xformers installation path
    xformers_path = None
    for path in site.getsitepackages() + [site.getusersitepackages()]:
        if path and os.path.exists(path):
            xformers_dirs = glob.glob(os.path.join(path, 'xformers*'))
            if xformers_dirs:
                xformers_path = xformers_dirs[0]
                break

    if not xformers_path:
        print("ERROR: xformers package not found")
        return False

    dispatch_file = os.path.join(xformers_path, 'ops', 'fmha', 'dispatch.py')
    if not os.path.exists(dispatch_file):
        print(f"ERROR: dispatch.py not found at {dispatch_file}")
        return False

    try:
        # Read the file
        with open(dispatch_file, 'r') as f:
            content = f.read()

        # Check if the file needs patching
        if '_USE_FLASH_ATTENTION_3 = True' in content:
            # Replace the flash attention 3 setting
            content = content.replace('_USE_FLASH_ATTENTION_3 = True', '_USE_FLASH_ATTENTION_3 = False')

            # Write the modified content back
            with open(dispatch_file, 'w') as f:
                f.write(content)

            print(f"SUCCESS: Patched {dispatch_file} - disabled flash attention 3")
            return True
        elif '_USE_FLASH_ATTENTION_3 = False' in content:
            print(f"INFO: {dispatch_file} already has flash attention 3 disabled")
            return True
        else:
            print(f"WARNING: Could not find _USE_FLASH_ATTENTION_3 variable in {dispatch_file}")
            return False

    except Exception as e:
        print(f"ERROR: Failed to patch {dispatch_file}: {e}")
        return False


if __name__ == "__main__":
    success = patch_xformers()
    sys.exit(0 if success else 1)
