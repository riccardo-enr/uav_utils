import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
control_path = os.path.join(dir_path, "../control")
control_path = os.path.realpath(control_path)
lib_path = os.path.join(control_path, "../")
lib_path = os.path.realpath(lib_path)
sys.path.append(lib_path)

print("dir_path: ", dir_path)
print("control_path: ", control_path)
print("lib_path: ", lib_path)

from control import nmpc_utils
