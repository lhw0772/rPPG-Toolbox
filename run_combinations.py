import yaml
from itertools import product
import subprocess


yaml_file = 'configs/infer_configs/PURE_MMPD_PHYSNET_BASIC.yaml'

temp_yaml_file = 'temp.yaml'
# Load the YAML file
with open(yaml_file, 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# Define the values for each scale key
scales = {
    'B_SCALE': [1.0, 2.0, 3.0],
    'S_SCALE': [1.0, 2.0, 3.0],
    'V_SCALE': [1.0, 2.0, 3.0],
    'SM_SCALE': [0.1, 0.5, 1.0],
    'FC_SCALE': [1.0, 5.0, 10.0]
}

combinations = list(product(*scales.values()))

for combo in combinations:
    for scale_key, value in zip(scales.keys(), combo):
        data['ADAPTER']['TENT'][scale_key] = value

    # Save the modified data back to the YAML file
    with open(temp_yaml_file, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

    # Run your desired Python file with the modified YAML
    subprocess.run(['python', 'main.py', '--config_file', temp_yaml_file])

