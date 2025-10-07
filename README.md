<div align="center">

# 🧠 Synthetic Human Face Generation using GANs
### Deep Learning Project for Realistic Face Synthesis

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*Generating realistic human faces from random noise using Generative Adversarial Networks*

[Features](#-features) • [Architecture](#-model-architecture) • [Installation](#-quick-start) • [Demo](#-streamlit-applications) • [Results](#-results--observations)

</div>

---

## 🎯 Project Overview

This repository showcases a **Deep Learning project** that leverages **Generative Adversarial Networks (GANs)** to synthesize photorealistic human faces from random noise vectors.

<table>
<tr>
<td width="50%">

### 🧠 Generator Network
Learns to produce synthetic images that are indistinguishable from real human faces

</td>
<td width="50%">

### 🔍 Discriminator Network
Learns to distinguish between authentic images and generated ones

</td>
</tr>
</table>

> **Adversarial Training Philosophy:** Through competitive training, both networks continuously improve — the generator becomes better at creating realistic faces, while the discriminator becomes more skilled at detection. This adversarial process results in highly convincing synthetic faces.

---

## ✨ Features

<table>
<tr>
<td>

- 🎨 **Generate Synthetic Faces** from random noise
- 🔍 **Real vs Fake Classification** using trained discriminator
- 📊 **Interactive Streamlit Apps** for easy experimentation
- 🧮 **Complete Training Pipeline** with visualization
- 💾 **Model Persistence** using Joblib for lightweight deployment
- 📈 **Training Metrics & Loss Curves** for monitoring

</td>
</tr>
</table>

---


## 🏗️ Model Architecture

<div align="center">

### Generator Architecture

</div>

| Component | Details |
|-----------|---------|
| **Input** | Random noise vector (latent dimension = 100) |
| **Hidden Layers** | Dense → Reshape → Conv2D/Conv2DTranspose layers |
| **Normalization** | Batch Normalization |
| **Activation** | LeakyReLU (hidden), tanh (output) |
| **Output** | Synthetic RGB image (128×128×3) |

<div align="center">

### Discriminator Architecture

</div>

| Component | Details |
|-----------|---------|
| **Input** | Real or generated image (128×128×3) |
| **Hidden Layers** | Multiple Conv2D layers with stride 2 |
| **Normalization** | Batch Normalization |
| **Activation** | LeakyReLU (hidden), sigmoid (output) |
| **Output** | Probability score (0 = fake, 1 = real) |

---

## 🧰 Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow, Keras |
| **Numerical Computing** | NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Image Processing** | OpenCV |
| **Web Interface** | Streamlit |
| **Utilities** | tqdm, Joblib |

</div>

---

## 📊 Training Configuration

<table>
<tr>
<td width="50%">

### Dataset & Preprocessing
- **Dataset:** Human face dataset  
  *(e.g., Face Mask Lite Dataset – Without Mask subset)*
- **Image Size:** 128×128 RGB
- **Preprocessing:** Normalization to [-1, 1]

</td>
<td width="50%">

### Hyperparameters
- **Noise Vector (z):** 100 dimensions
- **Optimizer:** RMSProp (lr = 0.0001)
- **Loss Function:** Binary Crossentropy
- **Epochs:** ~30
- **Batch Size:** 32

</td>
</tr>
</table>

### Training Process

\`\`\`
Epoch 1  → Generator learns basic shapes
Epoch 10 → Facial features start emerging
Epoch 20 → Realistic textures develop
Epoch 30 → High-quality synthetic faces
\`\`\`

> 📈 **Training Dynamics:** The generator attempts to fool the discriminator, while the discriminator learns to correctly classify real vs generated faces. Loss curves gradually converge, demonstrating adversarial learning stability.

---

## 🎨 Example Outputs

<div align="center">

### Generated Faces Across Training Epochs

| Epoch 10 | Epoch 20 | Epoch 30 |
|:--------:|:--------:|:--------:|
| ![gen3](assets/output3.png) | ![gen2](assets/output2.png) | ![gen1](assets/output1.png) |
| *Early features* | *Improved details* | *Photorealistic* |

*Generated samples progressively improve as the GAN learns complex visual features*

</div>

---

## 🚀 Streamlit Applications

### 🎨 Application 1: Image Generator

\`\`\`bash
streamlit run app3.py
\`\`\`

**Features:**
- 🎚️ Interactive slider to select number of generated images
- 🖼️ Real-time display of high-quality synthetic human faces
- 💾 Generated using the trained GAN model

---

### 🔍 Application 2: Real vs Fake Classifier

**Also available in `app3.py`:**

- 📤 Upload any JPG image
- 🤖 AI-powered prediction: Real or Synthetic
- 📊 Confidence score using the discriminator network

---

### 💡 Alternative: Lightweight Version

\`\`\`bash
streamlit run prediction.py
\`\`\`

> Uses Joblib-saved models for faster loading and deployment

---

## 🏆 Results & Observations

<table>
<tr>
<td>

### ✅ Achievements

- Generator successfully produces **realistic, smooth facial features** after sufficient training
- Discriminator stabilizes around **0.5 accuracy** — indicating balanced adversarial competition
- Model demonstrates strong **generalization** to unseen noise vectors

</td>
</tr>
<tr>
<td>

### ⚙️ Fine-tuning Potential

- Can be enhanced with **more training epochs**
- Performance improves with **larger face datasets** (e.g., CelebA, FFHQ)
- Architecture can be scaled for **higher resolution outputs**

</td>
</tr>
</table>

---

## 🚀 Future Improvements

<div align="center">

| Enhancement | Description |
|-------------|-------------|
| 🎯 **Progressive GANs** | Implement PGGAN or StyleGAN for higher-quality faces |
| 🔄 **Data Augmentation** | Apply advanced augmentation techniques |
| 🎓 **Transfer Learning** | Leverage pre-trained GAN architectures |
| 🌐 **Web Demo** | Build a public-facing web application |
| 📱 **Mobile App** | Create mobile interface for face generation |
| 🎭 **Conditional GANs** | Add control over facial attributes |

</div>


---

## 👨‍💻 Author
<div align="center">

<table>
<tr>
<td align="center">
<img src="https://github.com/Islamroubache.png" width="100px;" alt="Islam Roubache"/><br>
<sub><b>Islam Roubache</b></sub><br>
🎓 Master's Student in AI & Data Science<br>
📍 Higher School of Computer Science 08 May 1945<br>
Sidi Bel Abbes, Algeria
</td>
</tr>
</table>

</div>
<div align="center">


[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)

</div>

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**

Feel free to use, modify, and share with proper credit

[View License](LICENSE)

</div>

---

<div align="center">

### 💫 *"GANs don't just learn to generate data — they learn to imagine."*

---

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and TensorFlow

</div>
