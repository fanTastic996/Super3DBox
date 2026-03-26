<p align="center">
  <h1 align="center">🔲 SuperBox</h1>
  <p align="center"><b>Pose-Free 3D Perception through Emergent Global Consistency</b></p>
  <p align="center">
    <img src="https://img.shields.io/badge/status-work_in_progress-orange?style=for-the-badge" alt="Work in Progress"/>
    <img src="https://img.shields.io/badge/3D%20Vision-Implicit%20Localization-blue?style=for-the-badge" alt="3D Vision"/>
    <img src="https://img.shields.io/badge/Method-Slot%20Attention-purple?style=for-the-badge" alt="Slot Attention"/>
  </p>
</p>

---

### 🎯 The Challenge

Traditional implicit 3D localization methods hit a fundamental wall: **without explicit geometry** (camera extrinsics, calibrated poses), they cannot distinguish instances that share identical visual appearance yet occupy different spatial positions—like two visually identical chairs sitting apart in a room.

The standard recipe—injecting hard-coded 3D inductive biases—breaks down when camera parameters are unknown or noisy.

---

### 💡 Our Breakthrough

**SuperBox** introduces a radical departure: **we prove that explicit 3D priors are unnecessary.**

Instead, we harness **Global Contextual Consistency** emergent from pre-trained multi-view encoders (VGGT). Through **Global Competitive Masked Slot Attention**, our model achieves natural **Spatial Disentanglement** in a purely pose-free feature space.

> 🔥 **"Even when patch-level features collide, context-level competition separates."**

Even if two chairs exhibit nearly identical local semantic signatures at the patch level, our competitive mechanism sharply captures their distinct contextual relationships with surrounding structures—walls, windows, floor patterns—enabling precise 3D localization without ever seeing a camera matrix.

---

### 🌟 Key Highlights

| 🔍 **Zero Explicit Geometry** | 📐 **Emergent 3D Understanding** | ⚡ **Feed-Forward Inference** |
|:---:|:---:|:---:|
| No camera intrinsics or extrinsics required | Global consistency arises from foundation model priors | Direct 3D box prediction without iterative proposal generation |

---

### 🚧 Work in Progress

**SuperBox is currently under active development.** We are rigorously validating our approach on challenging multi-view datasets and refining the competitive attention mechanisms.

Stay tuned for code release, pre-trained checkpoints, and comprehensive benchmarks.

<p align="center">
  <i>🛠️ Building the future of pose-free 3D perception...</i>
</p>