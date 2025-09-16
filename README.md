# Welding Maintenance Demo (PyTorch)

Predictive maintenance for a robotic welding arm using camera vision and sensor signals.

## Setup
- Install dependencies (UV or pip):
  - uv sync
- Run help:
  - uv run python main.py --help

## Usage
- Train vision:
  - uv run python main.py train-vision
- Train sensor RF:
  - uv run python main.py train-sensor --model rf
- Train sensor LSTM:
  - uv run python main.py train-sensor --model lstm
- Predict on image:
  - uv run python main.py predict-vision --image path/to/img.jpg
- Predict on CSV:
  - uv run python main.py predict-sensor --model rf

Synthetic fallback data is used if real files are not present.
