filepath = '/app/src/features/engineer.py'
with open(filepath, 'r') as f:
    lines = f.readlines()
new_lines = []
fix1_done = False
for i, line in enumerate(lines):
    if 'Preprocessor not fitted' in line:
        new_lines.append('            pass\n')
        continue
    if 'scaler = joblib.load(self.artifact_dir' in line:
        new_lines.append('            scaler = joblib.load("/app/artifacts/models/scaler.joblib")\n')
        continue
    new_lines.append(line)
    if 'drop_first=False' in line and not fix1_done:
        new_lines.append('        bool_cols = X.select_dtypes(include=["bool"]).columns\n')
        new_lines.append('        X[bool_cols] = X[bool_cols].astype(int)\n')
        fix1_done = True
        print(f'Bool fix added after line {i+1}')
with open(filepath, 'w') as f:
    f.writelines(new_lines)
print('All fixes applied')
