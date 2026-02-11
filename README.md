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
```
## train
```bash
python train.py
```
Saves best model to: outputs/best_unet.keras

## Evaluate (Baseline)
```bash
python test.py
```
Baseline results (Test) after Train/Val split (90/10):

| Split |   Loss |   mIoU |   Dice |
| ----: |-------:|-------:|-------:|
|  Test | 0.3139 | 0.7023 | 0.8111 |


## Demo (Streamlit)
```bash
streamlit run app.py
```
## Qualitative Results

### Sample prediction (test set)
![pred0](assets/pred_0.png)

### Streamlit UI demo output
![ui](assets/ui_result.png)

## Project Structure

```md
- `train.py` training script
- `test.py` evaluation + saves prediction samples to `outputs/`
- `app.py` Streamlit inference UI
- `assets/` images used in README
- `outputs/` generated files (ignored by git)
```
