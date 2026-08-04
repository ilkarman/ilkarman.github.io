"""a quick stand-in dataset so the loop runs today. swap in the real gsm8k loader later.

emits simple 'a + b' problems in the gsm8k '#### <answer>' format so calculate_reward
works unchanged.
"""
import random


class _Dataset:
    def __init__(self, rows):
        self._rows = rows

    def shuffle(self, seed=0):
        rng = random.Random(seed)
        rows = self._rows[:]
        rng.shuffle(rows)
        return _Dataset(rows)

    def __getitem__(self, i):
        return self._rows[i % len(self._rows)]

    def __len__(self):
        return len(self._rows)


def make_arithmetic_dataset(n=512, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        rows.append({'question': f'What is {a} + {b}?', 'answer': f'#### {a + b}'})
    return _Dataset(rows)
