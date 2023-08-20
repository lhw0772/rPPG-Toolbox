import re
import glob

result_files = glob.glob("./log/*")

results = []

for file in result_files:
    with open(file, 'r') as f:
        content = f.read()

        mea_match = re.search(r'FFT MAE \(FFT Label\): (\d+\.\d+) \+/- (\d+\.\d+)', content)
        pearson_match = re.search(r'FFT Pearson \(FFT Label\): (\d+\.\d+) \+/- (\d+\.\d+)', content)

        if mea_match and pearson_match:
            mea_value = float(mea_match.group(1))
            pearson_value = float(pearson_match.group(1))
            filename = file

            results.append({
                'MEA': mea_value,
                'Pearson': pearson_value,
                'Filename': filename
            })

import matplotlib.pyplot as plt

mea_values = [result['MEA'] for result in results]
pearson_values = [result['Pearson'] for result in results]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(mea_values, bins=10, edgecolor='black')
plt.xlabel('MEA Value')
plt.ylabel('Frequency')
plt.title('Histogram of MEA Values')

plt.subplot(1, 2, 2)
plt.hist(pearson_values, bins=10, edgecolor='black')
plt.xlabel('Pearson Value')
plt.ylabel('Frequency')
plt.title('Histogram of Pearson Values')

plt.tight_layout()
plt.show()

min_mea_result = min(results, key=lambda x: x['MEA'])
max_pearson_result = max(results, key=lambda x: x['Pearson'])

print("Minimum MEA Value:")
print("MEA:", min_mea_result['MEA'])
print("Filename:", min_mea_result['Filename'])

print("\nMaximum Pearson Value:")
print("Pearson:", max_pearson_result['Pearson'])
print("Filename:", max_pearson_result['Filename'])