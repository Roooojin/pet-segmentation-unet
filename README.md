# Pet Segmentation with U-Net (Oxford-IIIT Pet)

TensorFlow + TFDS implementation of a simple U-Net for pet segmentation (trimap: background / pet / border).
Includes training, evaluation, and a Streamlit demo app.

## Dataset
Oxford-IIIT Pet via `tensorflow_datasets` (`oxford_iiit_pet`).

## Setup (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
