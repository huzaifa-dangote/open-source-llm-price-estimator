# Open Source LLM Price Estimator

A fine-tuned language model that estimates product prices from text descriptions using QLoRA (Quantized Low-Rank Adaptation) on Meta's Llama-3.2-3B model.

## Overview

This project fine-tunes an open-source language model to predict product prices based on text descriptions. The model is trained using QLoRA, an efficient fine-tuning technique that significantly reduces memory requirements while maintaining model performance.

## Features

- **Efficient Fine-tuning**: Uses QLoRA (4-bit quantization) to reduce memory footprint
- **Flexible Configuration**: Supports both LITE_MODE (for free T4 GPUs) and full mode (for A100 GPUs)
- **HuggingFace Integration**: Automatically pushes trained models to HuggingFace Hub
- **Weights & Biases Tracking**: Integrated with W&B for experiment tracking and monitoring
- **Evaluation Pipeline**: Includes comprehensive evaluation metrics

## Project Structure

```
open-source-llm-price-estimator/
├── QLoRA_Price_Estimator_Training.ipynb    # Training notebook
├── QLoRA_Price_Estimator_Model.ipynb        # Evaluation notebook
└── README.md
```

## Prerequisites

### Google Colab Setup

1. **Access the Notebooks**:
   - Open the training notebook: [QLoRA_Price_Estimator_Training.ipynb](https://colab.research.google.com/github/huzaifa-dangote/open-source-llm-price-estimator/blob/main/QLoRA_Price_Estimator_Training.ipynb)
   - Open the evaluation notebook: [QLoRA_Price_Estimator_Model.ipynb](https://colab.research.google.com/github/huzaifa-dangote/open-source-llm-price-estimator/blob/main/QLoRA_Price_Estimator_Model.ipynb)

2. **Required API Keys** (Store in Colab Secrets):
   - **HuggingFace Token**: Go to [HuggingFace Settings](https://huggingface.co/settings/tokens) and create a token
     - Add to Colab: `HF_TOKEN` in Colab Secrets (🔑 icon in sidebar)
   - **Weights & Biases API Key**: Get from [W&B Settings](https://wandb.ai/settings)
     - Add to Colab: `WANDB_API_KEY` in Colab Secrets

3. **GPU Requirements**:
   - **LITE_MODE=True**: Free T4 GPU (16GB) - suitable for smaller experiments
   - **LITE_MODE=False**: Paid A100 GPU (40GB+) - for full dataset training

## Quick Start

### 1. Training the Model

1. Open `QLoRA_Price_Estimator_Training.ipynb` in Google Colab
2. Configure your settings:
   - Set `HF_USER` to your HuggingFace username
   - Adjust `LITE_MODE` based on your GPU availability
   - Modify hyperparameters if needed (see Configuration section)
3. Run all cells sequentially
4. The model will be automatically saved to your HuggingFace Hub repository

### 2. Evaluating the Model

1. Open `QLoRA_Price_Estimator_Model.ipynb` in Google Colab
2. Update the constants:
   - Set `HF_USER` to your HuggingFace username
   - Update `RUN_NAME` to match your trained model's run name
   - Set `REVISION` to the specific commit hash if needed
3. Run all cells to evaluate the model on the test set

## Configuration

### Training Hyperparameters

The training notebook includes configurable hyperparameters:

```python
# Model Configuration
BASE_MODEL = "meta-llama/Llama-3.2-3B"
LITE_MODE = True  # Set to False for full dataset

# Training Parameters
EPOCHS = 3
BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4

# QLoRA Parameters
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.2
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

### LITE_MODE vs Full Mode

**LITE_MODE=True** (Recommended for free Colab):
- Uses smaller dataset subset
- Reduced validation set (500 samples)
- Lower batch size and LoRA rank
- Suitable for T4 GPU

**LITE_MODE=False** (For paid Colab):
- Uses full dataset
- Larger validation set (1000 samples)
- Higher batch size and LoRA rank
- Requires A100 GPU

## Model Architecture

- **Base Model**: Meta Llama-3.2-3B
- **Fine-tuning Method**: QLoRA (4-bit quantization)
- **LoRA Configuration**: 
  - Rank (r): 128
  - Alpha: 256
  - Target modules: Attention layers + MLP layers
- **Quantization**: 4-bit NF4 with double quantization

## Dataset

The project uses datasets from HuggingFace:
- **LITE_MODE**: `ed-donner/items_prompts_lite`
- **Full Mode**: `ed-donner/items_prompts_full`

The dataset contains product descriptions with corresponding price labels for training and evaluation.

## Training Process

1. **Data Loading**: Loads dataset from HuggingFace
2. **Model Initialization**: Loads base model with 4-bit quantization
3. **LoRA Setup**: Configures PEFT (Parameter-Efficient Fine-Tuning) with LoRA
4. **Training**: Uses SFTTrainer (Supervised Fine-Tuning) from TRL library
5. **Monitoring**: Logs metrics to Weights & Biases
6. **Model Saving**: Automatically pushes to HuggingFace Hub

## Evaluation

The evaluation notebook:
- Loads the fine-tuned model from HuggingFace Hub
- Tests on the test dataset
- Uses utility functions to compute evaluation metrics
- Provides performance insights

## Dependencies

Key libraries used:
- `transformers` - HuggingFace transformers library
- `bitsandbytes` - Quantization support
- `trl` - Transformer Reinforcement Learning library
- `peft` - Parameter-Efficient Fine-Tuning
- `datasets` - HuggingFace datasets
- `wandb` - Weights & Biases for tracking
- `torch` - PyTorch

All dependencies are automatically installed in the notebooks.

## Results

The model is trained to predict product prices from text descriptions. Evaluation metrics are computed using the utility functions provided in the evaluation notebook.

## Notes

- The model is saved as a private repository on HuggingFace Hub by default
- Training progress is logged to Weights & Biases (if `LOG_TO_WANDB=True`)
- Model checkpoints are saved periodically during training
- The evaluation notebook requires the specific run name and revision from your training session

## License

This project uses the Llama-3.2-3B model, which has its own licensing terms. Please review Meta's license agreement for Llama models.

## Author

**huzaifa-dangote**

## Acknowledgments

- Dataset provided by `ed-donner`
- Base model: Meta Llama-3.2-3B
- Built using HuggingFace Transformers, PEFT, and TRL libraries
