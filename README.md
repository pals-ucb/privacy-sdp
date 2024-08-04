# Application of ifferentially Private Stochastic Gradient Descent (DP-SGD) for LLM Privacy Protection (GPt2)

This repository focuses on implementing and experimenting with Differentially Private Stochastic Gradient Descent (DP-SGD) for training language models, specifically GPT-2, to ensure privacy protection during training. The work involves using both TensorFlow and PyTorch frameworks, with an emphasis on the latter due to compatibility and performance issues encountered with TensorFlow's Privacy library.

## Overview

### Purpose
The main goal of this project is to explore the application of DP-SGD in fine-tuning GPT-2 language models using the WikiText-2 dataset. This approach aims to provide strong privacy guarantees for the training data, preventing the model from memorizing sensitive information.

### Repository Layout
- **notebooks/**: Contains Jupyter notebooks used for experimental work and model training.
- **src/**: Source code for implementing DP-SGD and fine-tuning GPT-2.
- **papers/**: Collection of literature and white papers relevant to DP-SGD and privacy-preserving machine learning.
- **data/**: Scripts and instructions for downloading and preprocessing the WikiText-2 dataset.

## Experimental Work

### TensorFlow Implementation
Initial experiments were conducted using TensorFlow's Privacy library. However, due to dependency issues and compatibility problems, the focus was shifted to PyTorch and its Opacus library.

### PyTorch Implementation
Using the Opacus library, we successfully implemented DP-SGD for fine-tuning GPT-2 on the WikiText-2 dataset. The experiments demonstrated the feasibility of training large language models with differential privacy guarantees.

## Literature Survey

### Key Papers
- **"Pointer Sentinel Mixture Models"**: This paper introduces the WikiText-2 dataset, emphasizing its high quality and suitability for language modeling tasks.
- **Other relevant papers**: Included in the papers directory, providing foundational knowledge and advancements in differentially private machine learning.

## Final Project: Fine-Tuning GPT-2 with DP-SGD

### Approach
1. **Dataset Preparation**: Preprocessed the WikiText-2 dataset to be compatible with GPT-2 fine-tuning.
2. **Model Training**: Implemented DP-SGD using the Opacus library in PyTorch, fine-tuning GPT-2 on the preprocessed dataset.
3. **Evaluation**: Assessed the performance of the fine-tuned model, analyzing the trade-offs between privacy guarantees and model accuracy.

### Results
The final experiments showed that while DP-SGD introduces noise to ensure privacy, it is possible to achieve a reasonable balance between privacy and model performance. Detailed results and analysis can be found in the notebooks directory.

## Getting Started

### Prerequisites
- Python 3.7 or later
- PyTorch
- Opacus
- Other dependencies listed in `conda_env_requirements.txt`

### Installation
Clone the repository and install the required dependencies:
```sh
git clone https://github.com/pals-ucb/privacy-sdp.git
cd privacy-sdp
pip install -r conda_env_requirements/<select_appropriate_file>
```

### Running Experiments
1. **Data Preparation**: Download and preprocess the WikiText-2 dataset using the provided scripts in the `data` directory.
2. **Training**: Use the src/gpt2_main.py -h to see the different options


## Contributing
We welcome contributions to improve the project. Please submit pull requests or open issues for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
We thank the authors of the WikiText-2 dataset and the developers of TensorFlow Privacy and Opacus for their valuable tools and resources.

Special thanks to ChatGPT-4 by OpenAI for providing assistance with project on technical support, and documentation. Your contributions have been invaluable to the successful completion of this project.

```

This updated `README.md` includes the acknowledgement for ChatGPT-4's contributions. Let me know if there's anything else you'd like to add or modify!