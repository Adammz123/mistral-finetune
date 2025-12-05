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

    # Check multiple possible locations for dispatch.py
    dispatch_file = None
    
    for path in site.getsitepackages() + [site.getusersitepackages()]:
        if not path or not os.path.exists(path):
            continue
            
        # First try xformers-0.0.32.post2.dist-info
        candidate = os.path.join(path, 'xformers-0.0.32.post2.dist-info', 'ops', 'fmha', 'dispatch.py')
        if os.path.exists(candidate):
            dispatch_file = candidate
            print(f"INFO: Found dispatch.py in xformers-0.0.32.post2.dist-info")
            break
        
        # If not found, try xformers
        candidate = os.path.join(path, 'xformers', 'ops', 'fmha', 'dispatch.py')
        if os.path.exists(candidate):
            dispatch_file = candidate
            print(f"INFO: Found dispatch.py in xformers")
            break

    if not dispatch_file:
        print("ERROR: dispatch.py not found in either xformers-0.0.32.post2.dist-info or xformers")
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
