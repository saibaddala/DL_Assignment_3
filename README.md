Sure! Below is a clean and complete `README.md` file for your repository, combining both **Attention-based RNN (`Dl_Assignment_3_Attention`)** and **Vanilla RNN (`Dl_Assignment_3_no_attention`)** sections with correct project names and filenames. It's formatted professionally for GitHub and ready to use.

---

```markdown
# DL Assignment 3 — Telugu Transliteration using Seq2Seq Models

This repository contains two implementations for the Telugu transliteration task using the [Dakshina dataset](https://huggingface.co/datasets/dakshina):

1. **Vanilla RNN-based Seq2Seq model** (in `Dl_Assignment_3_no_attention`)
2. **Attention-based RNN Seq2Seq model** (in `Dl_Assignment_3_Attention`)

The objective is to convert Roman-script Telugu words to native Telugu script using deep learning models with encoder-decoder architecture.

---

## 📁 Folder Structure

```

├── Dl\_Assignment\_3\_no\_attention      # Vanilla RNN model (without attention)
├── Dl\_Assignment\_3\_Attention         # Attention-based RNN model
├── data/                             # Preprocessed train/val/test data (not included in repo)
├── predictions\_vanilla/             # Folder for storing test predictions from vanilla model
├── predictions\_attention/           # Folder for storing test predictions from attention model

````

---

## 🚀 Vanilla RNN Model (`Dl_Assignment_3_no_attention`)

### Code Overview

- **Libraries**: All necessary libraries are imported, including `torch`, `wandb`, etc.
- **WandB Integration**: Project name is set as `vanillaRNN`
- **GPU Support**: Automatically uses GPU if available

### Data Loading & Preprocessing

- Preprocessing functions:
  - `pre_processing(train_input, train_output)` — for training data
  - `pre_processing_validation(val_input, val_output)` — for validation/test data
- Custom dataset class: `MyDataset`
- DataLoader creation: `dataLoaderFun(...)`

### Training & Evaluation

- Training function: `train(...)`
  - Trains encoder and decoder with selected hyperparameters
  - Logs training loss, accuracy, validation loss, and accuracy at every epoch
- Accuracy calculation:
  - `validationAccuracy(...)` — computes validation loss and exact match accuracy

### Running the Script

- Script: `trainVanillaRNN.py`
- Run with:
  ```bash
  python trainVanillaRNN.py
````

* Custom parameters:

  ```bash
  python trainVanillaRNN.py --epochs 15 --cellType GRU --bidirection Yes --learnRate 1e-3
  ```

### Command-Line Arguments

```
-wp, --wandb_project        : WandB project name (default: vanillaRNN)
-es, --embSize              : Embedding size (default: 64)
-el, --encoderLayers        : Encoder layers (default: 5)
-dl, --decoderLayers        : Decoder layers (default: 5)
-hn, --hiddenLayerNuerons   : Hidden layer size (default: 512)
-ct, --cellType             : Cell type ['RNN', 'LSTM', 'GRU']
-bd, --bidirection          : Use bidirectional encoder ['Yes', 'no']
-d, --dropout               : Dropout rate (default: 0.3)
-nE, --epochs               : Number of epochs (default: 10)
-lR, --learnRate            : Learning rate (default: 1e-4)
-bS, --batchsize            : Batch size (default: 32)
-opt, --optimizer           : Optimizer ['Adam', 'Nadam']
-tf, --tf_ratio             : Teacher forcing ratio (default: 0.5)
```

### Jupyter Notebook

* `VanillaRNN.ipynb`:

  * Sweep configurations run using WandB
  * Can be run by replacing parser args with `wandb.config` values

---

## 🔍 Attention RNN Model (`Dl_Assignment_3_Attention`)

### Code Overview

* **WandB Integration**: Project name is set as `AttentionRNN`
* **GPU Support**: Automatically uses GPU if available

### Data Loading & Preprocessing

* Preprocessing functions:

  * `pre_processing(...)`
  * `pre_processing_validation(...)`
* Uses same `MyDataset` and `dataLoaderFun(...)` as vanilla version

### Training & Evaluation

* Model Components:

  * Encoder
  * Decoder
  * Attention module
* Training function: `train(...)`

  * Passes encoder hidden states to decoder along with attention
  * Logs attention weights
* Accuracy function: `validationAccuracy(...)` — same as vanilla model

### Running the Script

* Script: `trainAttentionRNN.py`
* Run with:

  ```bash
  python trainAttentionRNN.py
  ```
* You can modify parameters the same way as in vanilla model.

### Command-Line Arguments

Same flags as vanilla model, just `--wandb_project` should be `AttentionRNN`.

### Jupyter Notebook

* `AttentionRNN.ipynb`:

  * Run attention-based training via sweep configs
  * WandB logs show impact of cell type, attention, and layers

---

## ✅ How to Run

> Note: Make sure to activate your WandB account before running training or sweeps.

```bash
# Vanilla RNN training
cd Dl_Assignment_3_no_attention
python trainVanillaRNN.py

# Attention RNN training
cd Dl_Assignment_3_Attention
python trainAttentionRNN.py
```

---

## 📂 Output Folders

* `predictions_vanilla/`:

  * Contains `.txt` or `.csv` file with test predictions for vanilla RNN
* `predictions_attention/`:

  * Contains predictions from attention-based model

---

## 📊 Evaluation Metrics

* **Exact Match Accuracy**: Prediction is correct only if it exactly matches the ground truth.
* **Cross Entropy Loss**: Used as training and validation loss.
* **Additional Analyses** (optional):

  * Confusion matrices
  * Error trends by word length, frequency, character type

---

## 🧪 Dependencies

Make sure to install:

```bash
pip install torch torchvision torchaudio
pip install wandb
```

---

## 🔗 Credits

* Dataset: [Dakshina Dataset (Telugu)](https://huggingface.co/datasets/dakshina)
* Implemented as part of Deep Learning Assignment 3

```

Let me know if you'd like me to generate an actual table for sample predictions or embed WandB screenshots for your report or repo!
```
