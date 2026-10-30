"""
Generate synthetic sample data for testing the pipeline.
"""

import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

def generate_amp_sequence(length=20):
    n_cationic = max(3, int(length * 0.3))
    seq = []
    for i in range(length):
        if i % 2 == 0:
            seq.append(random.choice('KRFH'))
        else:
            seq.append(random.choice('WILF'))
    return ''.join(seq)

def generate_non_amp_sequence(length=20):
    seq = ''.join(random.choice('AGVSLDPET')
                 for i in range(length))
    return seq

def main():
    sequences = []
    labels = []
    mics = []

    for i in range(550):
        length = random.randint(10, 30)
        seq = generate_amp_sequence(length)
        sequences.append(seq)
        labels.append(1)
        mics.append(np.random.uniform(2, 60))

    for i in range(550):
        length = random.randint(10, 30)
        seq = generate_non_amp_sequence(length)
        sequences.append(seq)
        labels.append(0)
        mics.append(np.random.uniform(32, 512))

    for i in range(50):
        if i % 2 == 0:
            sequences.append(sequences[i])
            labels.append(labels[i])
            mics.append(mics[i])

    df = pd.DataFrame({
        'sequence': sequences,
        'label': labels,
        'mic': mics
    })

    df.to_csv('sample_data.csv', index=False)
    print("Generated %d sequences" % len(df))
    print("Class distribution: %s" % df['label'].value_counts().to_dict())
    print("Length distribution: %s" % df['sequence'].str.len().describe().round(1).to_dict())
    print("\nSample sequences:")
    print(df.head().to_string())

if __name__ == "__main__":
    main()
