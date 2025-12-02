🧠 MRI Tumor Detection — Model Documentation

This project includes three final trained models, each exported in different formats for compatibility with various deployment environments. Below is a clear explanation of each file, their purpose, and when you should use them.

📌 1. best_model.h5
✔ Purpose

This is the checkpoint model saved during training whenever the validation accuracy improved.
It represents the best-performing model during training, based purely on performance metrics (accuracy/loss).

✔ When to Use

When you want the most accurate model.

When running inference in TensorFlow/Keras-based environments.

For research, testing, and comparison.

✔ Good For

Highest accuracy

Reliable inference

Stable performance

📌 2. final_model.keras
✔ Purpose

This is the fully trained final version of the model saved in the new .keras format.
It includes the final architecture, weights, and training configuration.

✔ When to Use

When deploying on TensorFlow 2.12+ or future-proof systems.

For modern production codebases using .keras as the recommended format.

When you want easier editing, re-training, or exporting to other formats (TFLite, ONNX).

✔ Good For

Production deployment

Re-training

Conversions (e.g., TFLite / ONNX)

📌 3. model_arch.json + model_weights.h5

These two files together represent the model architecture and weights saved separately.

✔ Purpose

model_arch.json → Stores the neural network structure

model_weights.h5 → Stores the weights only

This approach was used previously in Keras for models where architecture and weights needed to be stored independently.

✔ When to Use

When you want to manually reconstruct the model.

When deploying in minimal environments that load architecture + weights separately.

When doing debugging or experimenting with modifications in architecture file.

✔ Good For

Lightweight deployment

Manual loading

Editing architecture files
