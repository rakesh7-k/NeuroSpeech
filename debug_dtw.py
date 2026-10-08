from dtw_classifier import load_calibration_data, dtw_distance

cal = load_calibration_data('calibration.json')
seq = cal['YES'][0]
results = []
for phrase, samples in cal.items():
    for sample in samples:
        results.append((phrase, dtw_distance(seq, sample)))
results.sort(key=lambda x: x[1])
print(results[:10])
