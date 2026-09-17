import numpy as np

from ai_noise_cancel_dtln.dataset import SyntheticNoiseDataset
from ai_noise_cancel_dtln.inference import StreamingEnhancer
from ai_noise_cancel_dtln.model import DTLN


def main():
    dataset = SyntheticNoiseDataset(num_samples=1, sample_length=16000)
    noisy, clean = dataset[0]

    enhancer = StreamingEnhancer(model=DTLN())
    output = enhancer.process(noisy.numpy())

    print(f"Noisy sample length: {len(noisy)}")
    print(f"Enhanced sample length: {len(output)}")
    print(f"Clean sample length: {len(clean)}")
    print(f"Mean output magnitude: {float(np.abs(output).mean()):.4f}")


if __name__ == "__main__":
    main()
