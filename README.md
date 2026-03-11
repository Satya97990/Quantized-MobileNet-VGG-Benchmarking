Edge AI Model Optimization: Rock-Paper-Scissors
This repository contains a comprehensive analysis of deep learning model optimization for Edge AI deployment, comparing MobileNetV2 and VGG16 architectures using the Rock-Paper-Scissors dataset.

Technique            Model            Size / Compression       Performance
Baseline (FP32)      VGG16            80.64 MB (1.0x)          96.51% Accuracy
Baseline (FP32)      MobileNetV2      8.63 MB (1.0x)           84.41% Accuracy
TFLite INT8          MobileNetV2      2.59 MB (~3.3x)          86.88% Accuracy
4-Bit Clustering     MobileNetV2     ~1.08 MB (8.0x)           0.00345 MSE

🛠️ Key Features
Architecture Comparison: Benchmarking the parameter-heavy VGG16 against the mobile-optimized MobileNetV2.

Post-Training Quantization: Utilizing the TFLite converter to transform models into INT8 format, achieving significant latency reductions (e.g., 15.52 ms/image for MobileNetV2).

N-Bit Weight Clustering: Implementation of K-Means clustering (4-bit) to reduce theoretical weight size by 8x.

Sensitivity Analysis: Identification of Depthwise Convolutional layers as the most vulnerable components to quantization and clustering distortion.

🔍 Critical Insights
VGG16 provides superior accuracy but is roughly 10x larger than MobileNetV2, making it less suitable for memory-constrained devices.

MobileNetV2 is the optimal candidate for Edge deployment, meeting the 75% accuracy threshold while maintaining a tiny footprint.

Depthwise layers showed the highest reconstruction errors (MSE), suggesting they require careful calibration or higher precision compared to standard layers.

🚀 Getting Started
Environment: Optimized for TensorFlow 2.x with GPU acceleration (e.g., Google Colab T4).

Dataset: Automatically sourced via tensorflow_datasets (rock_paper_scissors).

Run: Open Assignment2_Edge_AI.ipynb to execute the training, benchmarking, and quantization pipelines.

requirements.txt

# Core Machine Learning
tensorflow>=2.15.0
tensorflow-datasets>=4.9.0

# Data Manipulation & Analysis
pandas>=2.0.0
numpy>=1.23.5
scikit-learn>=1.2.0

# Visualization
matplotlib>=3.7.0

📂 How to use this in your Repo
Create the file: Create a new file named requirements.txt in your root directory.

Add the content: Copy and paste the block above.

Install: Users can then set up their environment by running:
pip install -r requirements.txt

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This project was completed as part of the Edge AI (CP 330) course assignment at the Indian Institute of Science (IISc), Bangalore.

The work involved:

Architecture Benchmarking: Comparing the efficiency and accuracy of MobileNetV2 and VGG16.

Model Optimization: Implementing INT8 quantization and 4-bit weight clustering to reduce memory footprint for edge deployment.

Performance Analysis: Achieving 86.88% accuracy with a compressed 2.59 MB MobileNetV2 model, successfully exceeding the assignment's 75% accuracy threshold.

