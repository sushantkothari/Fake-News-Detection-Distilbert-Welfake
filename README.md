# Fake News Detection Using DistilBERT on the WELFake Dataset

This repository provides a production-quality and research-aligned implementation of a Transformer-based fake news detection system. The project is designed with a strong emphasis on model architecture, training stability, and robustness validation. A pretrained DistilBERT encoder is fine-tuned on the WELFake benchmark dataset to perform binary classification of news articles into fake and real categories.

The implementation follows industry-standard and research-grade practices, making it suitable for academic benchmarking, applied machine learning research, and real-world deployment scenarios.

## Dataset Overview

All experiments are conducted using the WELFake (Word Embedding over Linguistic Features) dataset, a large-scale benchmark corpus for fake news detection.

Dataset URL: https://zenodo.org/records/4561253  
DOI: 10.5281/zenodo.4561253  
Publisher: Zenodo  
License: Creative Commons Attribution 4.0 International (CC BY 4.0)  
Published: February 25, 2021  

The dataset is constructed by merging multiple established fake news corpora, including Kaggle, McIntire, Reuters, and BuzzFeed Political datasets. This aggregation reduces dataset-specific bias and improves model generalization across heterogeneous news sources and writing styles.

After preprocessing and validation, the dataset contains 72,134 news articles with a near-balanced class distribution. Fake news samples are labeled as 0, while real news samples are labeled as 1.

## System Architecture Overview

The system follows a modular architecture consisting of text preprocessing, tokenization and encoding, Transformer-based classification, evaluation, and robustness validation. The architectural design prioritizes semantic representation quality, computational efficiency, and stability under constrained inference conditions.

## Transformer Backbone: DistilBERT

The core model architecture is based on the `distilbert-base-uncased` Transformer encoder. DistilBERT is a distilled variant of BERT that retains most of the representational power of the original model while significantly reducing parameter count and inference latency.

DistilBERT is selected for the following architectural advantages:
- Reduced model size and faster inference compared to BERT
- Lower memory footprint suitable for scalable training
- Strong empirical performance on sequence classification tasks
- Efficient fine-tuning on large textual corpora

The encoder processes tokenized input through stacked self-attention layers, producing deeply contextualized embeddings that capture semantic, syntactic, and discourse-level information critical for fake news detection.

## Classification Head Design

A sequence classification head is attached to the DistilBERT encoder using the HuggingFace `AutoModelForSequenceClassification` abstraction.

Architectural details:
- The final hidden representation corresponding to the [CLS] token is used as the pooled sequence embedding
- A fully connected linear layer maps this representation to two output logits
- Softmax normalization converts logits into class probabilities for fake and real news
- Cross-entropy loss is used as the optimization objective

This separation between the pretrained encoder and task-specific head ensures stable fine-tuning and architectural clarity.

## Tokenization and Input Representation

Tokenization is performed using the HuggingFace `AutoTokenizer` aligned with DistilBERT.

Configuration details:
- Maximum sequence length: 256 tokens
- Truncation enabled for long articles
- Fixed-length padding for batch consistency
- Attention masks applied to distinguish valid tokens from padding

This configuration balances contextual coverage with computational efficiency and stable batching.

## Training Architecture and Optimization Strategy

Model training is implemented using the HuggingFace `Trainer` framework, providing a structured, reproducible training loop with built-in evaluation and checkpointing.

The dataset is split using stratified sampling into training, validation, and test sets with an 80-10-10 ratio.

Optimization configuration:
- Optimizer: AdamW
- Learning rate: 2e-5
- Weight decay: 0.01
- Batch size: 16
- Number of epochs: 3
- Evaluation frequency: End of each epoch
- Model selection criterion: Validation F1-score

The best-performing checkpoint is automatically selected based on F1-score to ensure balanced precision and recall.

## Evaluation and Validation Architecture

Model evaluation is performed on a held-out test set that remains unseen during training and validation.

Reported metrics include:
- Accuracy
- F1-score
- Confusion matrix
- ROC–AUC
- Softmax probability confidence analysis

This evaluation strategy provides a comprehensive and unbiased assessment of model performance and generalization.

## Robustness Stress Testing

Architectural robustness is evaluated through a controlled stress test in which the maximum token context length is reduced from 256 to 128 tokens. This simulates constrained inference environments such as truncated inputs or resource-limited deployment scenarios.

A new DistilBERT model is trained from scratch under the reduced context configuration. Performance degradation is measured relative to the baseline model.

The stress-tested model maintains an F1-score above 0.97, with less than 0.5 percent relative performance drop. Assertion-based validation enforces minimum robustness guarantees, confirming architectural stability.

## Error and Confidence Analysis

Post-training analysis includes systematic inspection of misclassified samples to identify potential failure modes. Confusion matrix analysis confirms balanced error distribution across fake and real classes.

Softmax probability outputs are analyzed to evaluate confidence separation. A strong confidence gap is observed, indicating well-defined decision boundaries and low ambiguity in predictions.

## Usage

The complete preprocessing pipeline, model architecture, training configuration, evaluation logic, and robustness experiments are implemented in a single Jupyter Notebook. Users can clone the repository, place the WELFake dataset in the expected path, and execute the notebook sequentially to reproduce all experiments and results.

The repository follows standard GitHub conventions and includes this README and a LICENSE file.

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

Verma, P. K., Agrawal, P., and Prodan, R. WELFake: Word Embedding Over Linguistic Features for Fake News Detection. IEEE Transactions on Computational Social Systems, 2021. DOI: 10.1109/TCSS.2021.3068519. Dataset DOI: 10.5281/zenodo.4561253.
