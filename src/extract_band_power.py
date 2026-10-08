"""Extract and validate Figshare band-power arrays."""

from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import mne

from feature_extraction import BANDS, FEATURE_CHANNELS, condition_band_power


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "figshare" / "clean"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "figshare" / "bandpower"


def save_subject_band_power(input_path: Path) -> Path:
    """Extract condition-separated band power for one cleaned subject."""
    epochs = mne.read_epochs(input_path, preload=True, verbose=False)
    powers = condition_band_power(epochs)
    output_path = OUTPUT_ROOT / input_path.name.replace("-clean-epo.fif", "-bandpower.npz")
    np.savez_compressed(
        output_path,
        **{
            f"{condition}_{band}": values
            for condition, condition_values in powers.items()
            for band, values in condition_values.items()
        },
        channel_names=np.asarray(FEATURE_CHANNELS),
        sfreq=np.asarray(epochs.info["sfreq"]),
        bands=np.asarray(list(BANDS.keys())),
    )
    return output_path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_ROOT.glob("sub-*-clean-epo.fif"))
    if len(input_files) != 50:
        raise RuntimeError(f"Expected 50 clean FIF files, found {len(input_files)}")
    for input_path in input_files:
        save_subject_band_power(input_path)

    report_path = PROJECT_ROOT / "results" / "metrics" / "figshare_bandpower_summary.txt"
    report_path.write_text(
        "Figshare band-power extraction\n"
        f"Bands: {BANDS}\n"
        "Method: Welch PSD, n_fft=256, mean power within each band\n"
        f"Channels: {FEATURE_CHANNELS}\n"
        "Conditions saved separately: all, left_hand, right_hand\n"
        f"Subjects processed: {len(input_files)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()