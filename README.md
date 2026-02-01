# Fake News Detection Using DistilBERT on the WELFake Dataset

This repository contains a production-quality and research-aligned implementation of a Transformer-based fake news detection system. The project focuses on architectural soundness, training stability, and robustness validation while leveraging modern NLP best practices. A pretrained DistilBERT encoder is fine-tuned on the WELFake benchmark dataset to perform binary classification of news articles into fake and real categories.

The implementation emphasizes clarity, reproducibility, and empirical rigor, making it suitable for academic benchmarking, applied research, and real-world deployment scenarios.

## Dataset Overview

All experiments are conducted on the WELFake (Word Embedding over Linguistic Features) dataset, a large-scale, curated benchmark for fake news detection.

Dataset URL: https://zenodo.org/records/4561253  
DOI: 10.5281/zenodo.4561253  
Publisher: Zenodo  
License: Creative Commons Attribution 4.0 International (CC BY 4.0)  
Published: February 25, 2021  

The dataset is constructed by merging multiple established fake news corpora, including Kaggle, McIntire, Reuters, and BuzzFeed Political datasets. This design choice reduces dataset-specific bias and improves model generalization across heterogeneous news sources and writing styles.

After preprocessing and validation, the dataset contains 72,134 news articles with a near-balanced class distribution. Fake news samples are labeled as 0, while real news samples are labeled as 1.

## System Architecture Overview

The system follows a modular architecture consisting of four primary stages: text preprocessing, tokenization and encoding, Transformer-based classification, and evaluation with robustness validation.

The architectural design prioritizes semantic representation quality, computational efficiency, and stability under constrained inference conditions.

## Transformer Backbone: DistilBERT

At the core of the model is the `distilbert-base-uncased` Transformer encoder. DistilBERT is a distilled variant of BERT that preserves most of the representational power of the original model while significantly reducing parameter count and inference latency.

DistilBERT is selected for the following architectural advantages:
- Approximately 40 percent fewer parameters than BERT
- Faster fine-tuning and inference
- Lower memory footprint suitable for scalable training
- Strong empirical performance on sequence classification tasks

The encoder processes tokenized input sequences through stacked self-attention layers, generating deeply contextualized embeddings that capture semantic, syntactic, and discourse-level information essential for fake news detection.

## Classification Head Design

A sequence classification head is attached to the DistilBERT encoder using the HuggingFace `AutoModelForSequenceClassification` abstraction.

Architectural details:
- The final hidden representation corresponding to the [CLS] token is used as a pooled sequence embedding
- A fully connected linear layer maps the pooled representation to two output logits
- Softmax normalization converts logits into class probabilities for fake and real news
- Cross-entropy loss is used as the optimization objective

This design ensures a clean separation between the pretrained language model and the task-specific classification layer, enabling stable fine-tuning and interpretability.

## Tokenization and Input Representation

Text tokenization is performed using the HuggingFace `AutoTokenizer` aligned with DistilBERT. The tokenization strategy is designed to balance contextual coverage with computational efficiency.

Key configuration parameters:
- Maximum sequence length: 256 tokens
- Truncation enabled to handle long articles
- Fixed-length padding for consistent batch shapes
- Attention masks applied to distinguish padding tokens from valid input

This setup ensures efficient batching on GPU hardware while preserving sufficient semantic context for reliable classification.

## Training Architecture and Optimization Strategy

Model training is implemented using the HuggingFace `Trainer` framework, providing a structured and reproducible training loop with built-in evaluation and checkpointing.

The dataset is split using stratified sampling into training, validation, and test sets with an 80-10-10 ratio, preserving class balance across splits.

Optimization configuration:
- Optimizer: AdamW
- Learning rate: 2e-5
- Weight decay: 0.01
- Batch size: 16
- Number of epochs: 3
- Evaluation frequency: End of each epoch
- Model selection criterion: Validation F1-score

The best-performing checkpoint is automatically selected based on F1-score, ensuring balanced precision and recall across both classes.

## Evaluation and Validation Architecture

The evaluation pipeline is designed to provide a comprehensive and unbiased assessment of model performance.

Metrics reported:
- Accuracy
- F1-score
- Confusion matrix
- ROC–AUC
- Softmax probability confidence analysis

All metrics are computed on a held-out test set that remains unseen during training and validation, ensuring reliable generalization estimates.

## Robustness Stress Testing

Architectural robustness is evaluated through a controlled stress test that reduces the maximum token context length from 256 to 128 tokens. This simulates constrained inference environments such as truncated inputs or resource-limited deployments.

A new DistilBERT model is trained from scratch under the reduced context configuration. Performance degradation is quantitatively measured relative to the baseline model.

The stress-tested architecture maintains an F1-score above 0.97, with less than 0.5 percent relative performance drop. Assertion-based validation enforces minimum robustness guarantees, confirming the model’s resilience to context reduction.

## Error and Confidence Analysis

Post-training analysis includes systematic inspection of misclassified samples to identify potential failure patterns. Confusion matrix analysis confirms balanced error distribution across fake and real classes.

Softmax probability outputs are analyzed to assess confidence separation between predictions. A strong confidence gap is observed, indicating well-defined decision boundaries and low ambiguity in classification outcomes.

## Usage

The complete architecture, preprocessing pipeline, training configuration, evaluation procedures, and robustness experiments are implemented in a single Jupyter Notebook. Users can clone the repository, place the WELFake dataset in the specified path, and execute the notebook sequentially to reproduce all reported results.

The repository follows standard GitHub conventions and includes this README and a LICENSE file for documentation and compliance.

## Technology Stack

- Python
- NumPy
- Pandas
- NLTK
- Matplotlib
- Seaborn
- Scikit-learn
- HuggingFace Transformers
- HuggingFace Datasets
- Evaluate
- PyTorch

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

Verma, P. K., Agrawal, P., and Prodan, R. WELFake: Word Embedding Over Linguistic Features for Fake News Detection. IEEE Transactions on Computational Social Systems, 2021. DOI: 10.1109/TCSS.2021.3068519. Dataset DOI: 10.5281/zenodo.4561253.
