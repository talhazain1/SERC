# Graph-Based Emotion Recognition Using Audio Features

## Overview
This project implements a novel approach for emotion recognition from audio data using Graph Convolutional Networks (GCNs). The system leverages the structural advantages of graphs to model the temporal and spectral relationships within audio signals, enabling accurate classification into four emotion categories: `happy`, `sad`, `neutral`, and `angry`.

## Features
- **Audio Preprocessing**: Feature extraction from audio files (MP3 and FLAC formats) using Librosa.
- **Graph Construction**: Conversion of audio features into graph representations with temporal and spectral relationships.
- **GCN-Based Classification**: Implementation of a GCN architecture to classify emotions from the graph-structured data.
- **Visualization**: Comprehensive visualizations of training performance, confusion matrices, and system workflow.

## Dataset
This project uses two datasets:
1. **MP3 Dataset**: Contains compressed audio files.
2. **FLAC Dataset**: Consists of high-quality, lossless audio files.

Both datasets include labeled transcripts with four emotion categories: `happy`, `sad`, `neutral`, and `angry`. Preprocessing includes extracting features such as MFCCs and chroma features.

## Requirements
The project requires the following libraries:

- Python >= 3.7
- Torch
- PyTorch Geometric
- NetworkX
- Librosa
- Matplotlib
- NumPy
- Pandas

Install the required packages with:
```bash
pip install -r requirements.txt
```

## Workflow
1. **Feature Extraction**: Extract spectral features like MFCCs using Librosa.
2. **Graph Construction**: Build graphs where:
   - Nodes represent audio frames.
   - Edges represent temporal relationships.
3. **GCN Classification**:
   - Two GCN layers aggregate graph features.
   - Fully connected layers map features to output classes.
4. **Training and Evaluation**:
   - Train the model on the graph dataset.
   - Evaluate using accuracy, precision, recall, and F1-score.
5. **Visualization**:
   - Loss/accuracy curves.
   - Confusion matrix.
   - Workflow and architecture diagrams.

## Project Structure
```
.
├── data/
│   ├── mp3_graphs/
│   ├── flac_graphs/
│   └── labeled_transcripts.csv
├── models/
│   └── gcn_model.py
├── notebooks/
│   └── preprocessing.ipynb
├── scripts/
│   ├── graph_construction.py
│   ├── train_gcn.py
│   └── evaluate.py
├── visuals/
│   ├── accuracy_loss_curves.png
│   ├── confusion_matrix.png
│   └── workflow_diagram.png
└── README.md
```

## Results
- **Accuracy**:
  - MP3 Dataset: 78.6%
  - FLAC Dataset: 80.1%
- **Performance Metrics**: High precision and recall for dominant emotion classes.
- **Visualization**:
  - Training and validation accuracy curves.
  - Confusion matrices illustrating class-wise performance.
  - Workflow diagrams highlighting the methodology.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/username/graph-emotion-recognition.git
   cd graph-emotion-recognition
   ```
2. Extract audio features and construct graphs:
   ```bash
   python scripts/graph_construction.py
   ```
3. Train the GCN model:
   ```bash
   python scripts/train_gcn.py
   ```
4. Evaluate the model:
   ```bash
   python scripts/evaluate.py
   ```

## Future Work
- **Improved Graph Construction**: Exploring advanced methods to capture richer relationships.
- **Advanced Architectures**: Incorporating Graph Attention Networks (GATs) for better performance.
- **Multimodal Analysis**: Combining audio, text, and visual data for holistic emotion recognition.

## References
1. Kipf, T. N., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks.
2. Velickovic, P., et al. (2018). Graph Attention Networks.
3. Librosa: A Python package for audio and music analysis, [https://librosa.org](https://librosa.org).
4. Yang, Z. G., & Váradi, T. (2023). Training Experimental Language Models with Low Resources.
5. Develasco, M., et al. (2022). Automatic Analysis of Emotions from Voices/Speech in Spanish TV Debates.

---

For questions or contributions, please contact [talha.10.zain@gmail.com](mailto:talha.10.zain@gmail.com).
