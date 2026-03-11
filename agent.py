from kimi_utils import *
import json
import time
from string import Template
 
file_paths = Path('src/backends/cuda/').rglob('*.h', case_sensitive=False)
header_files = [str(i) for i in file_paths]
file_paths = Path('src/backends/cuda/').rglob('*.hpp', case_sensitive=False)
for i in file_paths:
    header_files.append(str(i))
    
file_paths = Path('src/backends/cuda/').rglob('*.cpp', case_sensitive=False)
cpp_paths = [str(i) for i in file_paths]
file_paths = Path('src/backends/cuda/').rglob('*.cu', case_sensitive=False)
cu_paths = [str(i) for i in file_paths]


header_template = Template(read_file('header.py'))
cpp_template = Template(read_file('cpp.py'))
cu_template = Template(read_file('cu.py'))

# HEADER already done

# for file in header_files:
#     run_process(header_template.substitute(FILE_PATH=file.replace('\\', '/')))
    
for file in cpp_paths:
    run_process(cpp_template.substitute(FILE_PATH=file.replace('\\', '/')))
    
for file in cu_paths:
    run_process(cu_template.substitute(FILE_PATH=file.replace('\\', '/')))