# Technical Proposal: The Kinematic-Structural Refinement Network (KSR-Net) for Robust Multi-Frame License Plate Recognition

## 1. Introduction and Problem Formulation

The 2026 ICPR Competition on Low-Resolution License Plate Recognition (LRLPR) presents a challenge that lies at the precise intersection of forensic video analysis, computer vision, and pattern recognition.1 The task—predicting license plate text from sequences of five consecutive low-resolution (LR), motion-blurred, and compressed images—is not merely an academic exercise in super-resolution. It is a simulation of the most pervasive bottleneck in modern Intelligent Transportation Systems (ITS) and law enforcement: the degradation of critical evidential data by the physical and hardware limitations of surveillance infrastructure.3

In real-world surveillance scenarios, license plates are frequently captured by wide-angle dashboard cameras or high-elevation CCTV units. These sensors often operate under suboptimal conditions, resulting in images that suffer from a triad of degradations: heavy quantization noise from video compression (e.g., H.264/H.265), severe motion blur due to high relative velocities between the vehicle and the shutter, and low spatial resolution caused by the distance of the subject.3 The objective of the LRLPR competition is to reconstruct the semantic content—the alphanumeric string—from these degraded inputs. Crucially, the competition provides high-resolution (HR) ground truth images only for training, requiring participants to develop models that can infer high-fidelity information from low-fidelity inputs during the inference phase.3

This report proposes the **Kinematic-Structural Refinement Network (KSR-Net)**, a novel deep learning architecture designed specifically for this competition. KSR-Net departs from traditional Single-Image Super-Resolution (SISR) approaches, which often fail to utilize the temporal redundancy present in video sequences.6 Instead, it adopts a Multi-Frame Super-Resolution (MFSR) paradigm that leverages the kinematic consistency of vehicle motion to aggregate sub-pixel information across frames. By integrating a hybrid alignment mechanism (coupling optical flow with deformable attention), a video-transformer backbone, and a task-driven semantic recognition head, KSR-Net aims to surpass the current state-of-the-art MF-LPR$^2$ framework 3 by maximizing information retrieval while strictly adhering to the forensic requirement of preserving evidential value.

### 1.1 The Forensic Constraint: Evidential Value vs. Hallucination

A defining characteristic of this challenge is the need for "evidential value".3 In general image restoration, Generative Adversarial Networks (GANs) have achieved remarkable success in producing visually pleasing results by hallucinating high-frequency textures based on learned priors. However, in License Plate Recognition (LPR), such hallucination is a critical failure mode. A generative model that reconstructs a blurry '8' as a sharp 'B' due to a learned distribution bias creates a "false positive" identification, which is unacceptable in legal and forensic contexts.3

Consequently, the proposed KSR-Net minimizes reliance on generative priors that might alter the semantic content. Instead, it focuses on **Signal Aggregation**. The core hypothesis is that the information required to resolve a character is present across the five input frames, distributed in the sub-pixel phase shifts caused by the vehicle's motion. The challenge is not to invent new pixels, but to accurately align and fuse the existing pixels to reveal the latent signal buried under quantization noise and blur.8 This aligns with the competition's emphasis on recognition accuracy over mere perceptual quality.1

### 1.2 Mathematical Formulation of the Degradation Model

To engineer a robust restoration pipeline, we must first mathematically model the degradation process governing the competition dataset. Let $Y \in \mathbb{R}^{H \times W \times C}$ be the latent high-resolution license plate image. The observed low-resolution sequence $X = \{x_1, x_2,..., x_T\}$ (where $T=5$ for this competition) is generated via a degradation function $\mathcal{D}$:

$$x_t = \mathcal{D}(Y, \theta_t) = \downarrow_s ( \mathcal{K}_t * \mathcal{W}_t(Y) ) + n_t$$

Where:

- $\mathcal{W}_t$ represents the geometric warping at time $t$, induced by the relative motion (translation, rotation, and perspective distortion) of the vehicle.
    
- $\mathcal{K}_t$ denotes the blur kernel, a composite of the camera's Point Spread Function (PSF) and the anisotropic motion blur vector caused by exposure time.
    
- $\downarrow_s$ is the downsampling operator with scale factor $s$ (typically 4x in such challenges).
    
- $n_t$ represents additive noise, including thermal sensor noise and quantization artifacts from lossy compression.
    

The goal of KSR-Net is to learn the inverse mapping $\mathcal{F}_{\theta}: X \to \hat{Y}$ such that the semantic distance between the recognized text on $\hat{Y}$ and the ground truth text is minimized. This formulation explicitly acknowledges that the "noise" $n_t$ is not merely Gaussian but structured (compression artifacts), and the warp $\mathcal{W}_t$ is not merely rigid but projective.9

---

## 2. Landscape Analysis: Reviewing the State-of-the-Art

Before detailing the KSR-Net architecture, it is essential to analyze the existing solution space, specifically focusing on the MF-LPR$^2$ framework referenced in the competition materials, and the broader context of Video Super-Resolution (VSR).

### 2.1 The Baseline: Analysis of MF-LPR$^2$

The MF-LPR$^2$ framework 3 serves as the primary benchmark for this domain. It introduced a paradigm shift by moving away from single-frame restoration towards a multi-frame approach that utilizes optical flow for alignment.

**Core Mechanism:** MF-LPR$^2$ employs a "Filter and Refine" strategy. It calculates optical flow between the reference frame and neighboring frames using a state-of-the-art estimator (FlowFormer++). Crucially, it recognizes that optical flow estimations are often erroneous in the presence of motion blur or specular highlights. To mitigate this, it implements a temporal filtering module that checks for forward-backward consistency. If the flow consistency error exceeds a threshold, the neighboring frame (or specific pixels within it) is discarded or down-weighted.3

**Strengths:**

- **Artifact Suppression:** By rejecting poorly aligned features, MF-LPR$^2$ avoids the "ghosting" artifacts that plague naive VSR methods, preserving the structural integrity of characters.6
    
- **Non-Generative:** It relies on pixel aggregation rather than hallucination, maintaining evidential value.3
    
- **Performance:** It achieves 86.44% recognition accuracy on the RLPR dataset, significantly outperforming single-frame methods (14.04%).3
    

Weaknesses and Opportunities:

The reliance on explicit optical flow and hard filtering is a double-edged sword. In scenarios of extreme degradation—where the LRLPR competition is likely focused—optical flow algorithms often fail catastrophically because the "brightness constancy" assumption (that a pixel's intensity remains constant as it moves) is violated by changing reflections on the metallic plate and atmospheric interference. When flow fails, MF-LPR$^2$ discards the frame. This results in data loss. If 3 out of 5 frames are blurry and yield poor flow, the model effectively reverts to a single- or dual-frame restorer, losing the benefits of MFSR.

KSR-Net improves upon this by introducing a **soft alignment mechanism**. Instead of discarding misaligned features, we employ Deformable Attention 12 to search for valid correlations even when explicit optical flow fails, ensuring that every available photon of information contributes to the reconstruction.

### 2.2 Evolution of Video Super-Resolution (VSR)

The broader VSR field provides the building blocks for KSR-Net. Early methods like VESPCN and SPMC relied on simple motion compensation. The advent of Transformers has revolutionized this space.

- VRT (Video Restoration Transformer) 13: VRT demonstrated that long-range temporal dependencies can be modeled using parallel frame processing and multi-scale attention. However, its high computational cost makes it difficult to tune for specific tasks like LPR.
    
- SwinIR and Video Swin 15: The Swin Transformer introduces shifted window attention, which provides an optimal balance between local feature extraction (crucial for character edges) and global context (crucial for plate layout). KSR-Net adopts the Swin architecture but adapts it for 3D spatio-temporal volumes.
    

### 2.3 Advances in Scene Text Recognition (STR)

The restoration module must feed into a recognition module. The traditional CRNN (Convolutional Recurrent Neural Network) approach is robust but limited in handling irregular text or severe occlusions.

- PARSeq (Permuted Autoregressive Sequence) 17: This represents the current SOTA in STR. Unlike standard autoregressive models that read left-to-right, PARSeq learns to predict characters using an ensemble of permutation masks (e.g., predicting the first character based on the last three). This bidirectional context is invaluable for license plates, where a speck of dirt might obscure the first digit, but the syntax of the plate (e.g., "3 letters, 4 numbers") allows the model to infer the likely category of the missing character. KSR-Net integrates PARSeq not just as a final classifier, but as a source of semantic supervision during training.18
    

---

## 3. The Proposed Architecture: Kinematic-Structural Refinement Network (KSR-Net)

KSR-Net is a unified, end-to-end differentiable framework designed to ingest a tuple of LR frames and output the recognized text string. It comprises four sequentially integrated modules:

1. **Hybrid Alignment Module (HAM):** Synergizing Optical Flow and Deformable Attention.
    
2. **Spatio-Temporal Fusion Transformer (STFT):** A 3D Video Swin backbone.
    
3. **Task-Driven Reconstruction Head:** Optimized with gradient and semantic losses.
    
4. **Semantic Recognition Head:** A fine-tuned PARSeq module.
    

### 3.1 Module 1: Hybrid Alignment Module (HAM)

Alignment is the cornerstone of MFSR. If the features from Frame $t-1$ are not spatially aligned with Frame $t$, the fusion operation will result in a blurrier image than the input. KSR-Net employs a "Coarse-to-Fine" hybrid strategy.

#### 3.1.1 Coarse Alignment via FlowFormer++

We utilize a distilled version of **FlowFormer++** 19 to estimate the dense motion field between the reference frame ($L_{ref}$) and each supporting frame ($L_i$). FlowFormer++ uses a transformer-based cost volume processing mechanism that is more robust to large displacements than traditional CNN-based flow estimators like PWC-Net.

The estimated flow $F_{i \to ref}$ allows us to warp $L_i$ to the reference coordinate system:

$$\tilde{L}_i = \mathcal{W}(L_i, F_{i \to ref})$$

However, as noted in the analysis of MF-LPR$^2$, this warped image $\tilde{L}_i$ suffers from artifacts where the flow is inaccurate (occlusions, specularities).

#### 3.1.2 Fine Correction via Flow-Guided Deformable Attention (FGDA)

To correct the residual errors in $\tilde{L}_i$, we introduce the **FGDA** block. Unlike standard Deformable Convolution (DCN) which blindly predicts offsets, FGDA utilizes the _residual error map_ of the optical flow to guide the attention mechanism.12

We compute the difference map $E_i = | \tilde{L}_i - L_{ref} |$. This map highlights regions where optical flow failed. We then concatenate the features of the warped frame, the reference frame, and the error map, and pass them into a lightweight offset predictor network:

$$\Delta P_i = \text{Conv}(\text{Concat}(\Phi(\tilde{L}_i), \Phi(L_{ref}), E_i))$$

The Deformable Attention mechanism then resamples features from $\tilde{L}_i$ using these learned offsets $\Delta P_i$.

$$\hat{F}_i = \text{DeformAttn}(\Phi(\tilde{L}_i), \Delta P_i)$$

**Insight:** This hybrid approach provides a safety net. In regions where optical flow is perfect (background, static edges), the error $E_i$ is near zero, and the offsets $\Delta P_i$ remain small. In regions of complex motion blur where flow fails, $E_i$ is large, prompting the network to learn larger offsets to "search" the neighborhood for the correct feature, effectively implementing a soft, learnable search window rather than the hard rejection of MF-LPR$^2$.12

### 3.2 Module 2: Spatio-Temporal Fusion Transformer (STFT)

Once the feature maps from the 5 frames are aligned, they must be fused to extract the high-resolution signal. We reject standard CNN backbones in favor of a **Video Swin Transformer** architecture 15, which treats the sequence as a 3D spatio-temporal volume.

#### 3.2.1 3D Shifted Window Attention

The core innovation of the Swin Transformer is Shifted Window Attention. In the context of video (3D data), we partition the feature volume $T \times H \times W$ into non-overlapping 3D windows of size $P_T \times P_H \times P_W$ (e.g., $2 \times 8 \times 8$).

Standard multi-head self-attention (MSA) is computed within each window. This captures local spatial texture and immediate temporal correlation (between the 2 frames in the window).

$$\text{Attention}(Q, K, V) = \text{Softmax}(\frac{QK^T}{\sqrt{d}} + B)V$$

Where $B$ is the relative position bias, crucial for encoding the spatial geometry of characters.

In the subsequent layer, the window partition is shifted by $(P_T/2, P_H/2, P_W/2)$. This "Shifted Window" configuration bridges the boundaries of the previous windows.

- **Spatial Bridge:** Information flows from the top-left quadrant of a character to the bottom-right.
    
- **Temporal Bridge:** Information flows from Frame 1-2 to Frame 2-3, eventually propagating across the entire 5-frame sequence.
    

This architecture allows KSR-Net to build a global understanding of the license plate structure while maintaining the computational efficiency of local windows, avoiding the quadratic complexity of global ViTs.15

### 3.3 Module 3: Task-Driven Reconstruction Head

The STFT outputs a fused, high-dimensional feature map. A series of PixelShuffle layers upscales this to the target resolution. However, the efficacy of this reconstruction is determined by the **loss landscape** used during training. We propose a composite loss function that prioritizes text legibility over pixel smoothness.

#### 3.3.1 The Loss Landscape

The total loss $\mathcal{L}_{total}$ is a weighted sum of four components:

$$\mathcal{L}_{total} = \lambda_{pix}\mathcal{L}_{Char} + \lambda_{grad}\mathcal{L}_{Edge} + \lambda_{percep}\mathcal{L}_{LPIPS} + \lambda_{sem}\mathcal{L}_{STR}$$

1. **Charbonnier Loss ($\mathcal{L}_{Char}$):** A robust variant of $L_1$ loss: $\sqrt{||I_{SR} - I_{GT}||^2 + \epsilon^2}$. Unlike $L_2$ (MSE), which penalizes large errors heavily and leads to blurry "average" images, Charbonnier is more forgiving of outliers and produces sharper edges.3
    
2. Gradient Profile Loss ($\mathcal{L}_{Edge}$): License plates are binary signals (text vs. background). Legibility is defined by the sharpness of the transition gradient. We compute the gradient magnitude maps of the SR and GT images and minimize their difference:
    
    $$ \mathcal{L}_{Edge} = |
    

| \nabla I_{SR} - \nabla I_{GT} ||1 $$

This forces the network to prioritize high-frequency boundaries.22

3. **LPIPS Perceptual Loss ($\mathcal{L}{LPIPS}$):** We use a pre-trained VGG network to extract deep features and minimize the distance in feature space. This encourages the recovery of realistic textures (e.g., the reflective sheen of the plate) that pixel losses miss.[3]

4. **Semantic Recognition Loss ($\mathcal{L}{STR}$):** This is the "Task-Driven" component.23 We pass the generated $I{SR}$ into a frozen, pre-trained text recognizer. We minimize the Cross-Entropy loss between the predicted text and the ground truth string.

* Mechanism: Backpropagating gradients from the recognizer through the super-resolution network forces the SR model to enhance features that are critical for machine readability (e.g., separating the stroke of an 'E' from an 'F'), even if those features don't minimize PSNR.

### 3.4 Module 4: Semantic Recognition Head (PARSeq)

The final component is the recognition head. We utilize **PARSeq** 18, which is currently the gold standard for robust text recognition.

**Architecture:** PARSeq employs a 12-layer Vision Transformer (ViT) encoder and a unified decoder that is trained with Permuted Language Modeling (PLM).

- **Robustness to Occlusion:** In standard autoregressive decoding (Left-to-Right), if the first character is occluded, the model's internal state is corrupted for all subsequent characters. PARSeq is trained to decode sequences in random orders (e.g., Right-to-Left, Middle-Out). During inference, it can ensemble these different decoding paths. If the first character is blurry, the Right-to-Left path (which has seen the clear end of the plate) provides a strong prior to resolve the ambiguity.
    
- **Fine-Tuning:** We initialize PARSeq with weights trained on massive synthetic datasets (MJSynth, SynthText) and fine-tune it on the competition's training set. We also employ a "validity filter" to restrict predictions to the alphanumeric set valid for license plates, suppressing special characters that often appear as noise artifacts.18
    

---

## 4. Dataset Strategy and Synthetic Augmentation

The quality of a deep learning model is bounded by the diversity of its training data. While the competition provides a training set (RLPR), relying solely on it is insufficient due to the domain gap between the training distribution and the "blind" test set. KSR-Net employs a rigorous **Data-Centric AI** strategy.

### 4.1 The RLPR Dataset and its Limitations

The RLPR dataset 3 contains 200 pairs of sequences. While high-quality, this is too small for training a Transformer-based model from scratch without overfitting. The degradation patterns in RLPR (real-world capture) are complex, involving non-linear camera response functions (CRF) and sensor noise.

### 4.2 The "Hard" Synthetic Degradation Pipeline

To augment the data, we create a synthetic generator that degrades high-resolution license plate images (from open datasets like CCPD or UFPR-ALPR) to match the competition's domain. We do _not_ use simple bicubic downsampling.25 Instead, we implement a physics-based degradation pipeline 26:

1. **Motion Blur Simulation:** We generate random motion trajectories (linear and non-linear kernels) to simulate vehicle vibration and shutter integration time. The kernel angles are sampled from a distribution matching the camera perspective (typically horizontal/diagonal).
    
2. **Atmospheric Turbulence:** We apply elastic distortions to simulate the "heat shimmer" often seen in long-range surveillance zoom.
    
3. **Compression Artifact Injection:** We encode image patches using actual JPEG and MPEG quantization tables at varying Quality Factors (Q=10 to Q=50). This introduces the specific $8 \times 8$ blocking artifacts that the network must learn to remove.3
    
4. **Sensor Noise Modeling:** We inject heteroscedastic Gaussian-Poisson noise, where the noise variance is dependent on signal intensity ($\sigma^2(y) = a \cdot y + b$), mimicking the photon shot noise of CMOS sensors in low light.
    

This synthetic data curriculum allows the model to see millions of unique degradation combinations, ensuring robust generalization to the unseen test set.25

### 4.3 Test-Time Augmentation (TTA)

During the inference phase (where no ground truth is available), we employ **Test-Time Augmentation** to boost performance.27 Theoretical analysis suggests that TTA reduces the variance of the prediction error by averaging out the stochasticity of the network's response to specific input noise patterns.29

For each input sequence $X$, we generate a set of augmented views $\mathcal{T}(X)$:

1. **Identity:** Original sequence.
    
2. **Flip:** Horizontally flipped sequence (with text labels reversed).
    
3. **Scale:** Resized sequences (0.9x, 1.1x) to handle scale variance.
    
4. **Gamma:** Brightness adjusted sequences ($\gamma=0.8, 1.2$).
    

We feed all views into KSR-Net. The final text prediction is obtained via **Logit Voting**: we sum the softmax probabilities from the PARSeq head for all views and select the character with the maximum aggregate probability. This ensemble approach effectively filters out transient recognition errors caused by specific local noise artifacts.30

---

## 5. Experimental Design and Ablation Studies

To validate the KSR-Net architecture, we propose a rigorous experimental protocol. The following table outlines the comparative analysis against the MF-LPR$^2$ baseline and variations of KSR-Net.

### 5.1 Comparative Metrics

We utilize two categories of metrics:

- **Restoration Metrics:** PSNR, SSIM, and LPIPS to measure visual fidelity.
    
- **Recognition Metrics:** Accuracy (Exact Match) and Character Error Rate (CER). Note that for the competition, Recognition Accuracy is the primary ranking criterion.
    

### 5.2 Ablation Framework

|**Model Variant**|**Alignment Strategy**|**Backbone**|**Loss Function**|**Hypothesis**|
|---|---|---|---|---|
|**Baseline (MF-LPR$^2$)**|Optical Flow + Filtering|CNN (ResNet-based)|Pixel + LPIPS|Strong baseline, but drops frames with bad flow.|
|**KSR-Net (V1)**|Optical Flow Only|**Video Swin**|Pixel + LPIPS|Tests the benefit of the Transformer backbone over CNN.|
|**KSR-Net (V2)**|**Hybrid (Flow + Deform)**|Video Swin|Pixel + LPIPS|Tests the benefit of the Hybrid Alignment (HAM).|
|**KSR-Net (Final)**|**Hybrid (Flow + Deform)**|Video Swin|**Pixel + Grad + Semantic**|Tests the benefit of Task-Driven Loss.|

Anticipated Results:

We hypothesize that KSR-Net (V1) will show improved PSNR due to the superior global context of Swin Transformers. KSR-Net (V2) is expected to show a significant jump in SSIM on the subset of the test set with high motion blur, as the Deformable Attention recovers information that Flow misses. Finally, KSR-Net (Final) should demonstrate the highest Recognition Accuracy, potentially at the cost of a slight drop in PSNR, as the Semantic Loss forces the network to create "hyper-legible" characters that may deviate slightly from the pixel-perfect ground truth but are easier for PARSeq to read.

---

## 6. Implementation Strategy

### 6.1 Hardware and Infrastructure

Training 3D Video Transformers is memory-intensive. The proposed infrastructure utilizes a cluster of **4x NVIDIA A100 (80GB)** GPUs.

- **Framework:** PyTorch 2.4 with CUDA 12.1.
    
- **Mixed Precision:** We employ Automatic Mixed Precision (AMP) with `bfloat16`. This provides the speed and memory savings of FP16 without the numerical instability often seen in Flow and Transformer training.19
    
- **Gradient Accumulation:** To simulate a large batch size (which stabilizes Transformer convergence) within GPU memory limits, we use gradient accumulation to achieve an effective batch size of 64 sequences.
    

### 6.2 Training Schedule

We adopt a **Three-Stage Training Protocol** to ensure stability:

- **Stage 1: Alignment Pre-training (20 epochs).** We train the HAM module alone on the FlyingChairs and MPI Sintel datasets to learn generic motion estimation. We then fine-tune it on the LPR dataset using unsupervised photometric loss. This ensures the backbone receives reasonable alignments from the start.
    
- **Stage 2: Restoration Warm-up (50 epochs).** We freeze the HAM and PARSeq head. We train the STFT backbone using only $\mathcal{L}_{Char}$ and $\mathcal{L}_{Edge}$. This allows the transformer to learn texture reconstruction without the noisy gradients from the uninitialized semantic head.
    
- **Stage 3: End-to-End Fine-tuning (100 epochs).** We unfreeze all modules and train with the full $\mathcal{L}_{total}$. We use a Cosine Annealing Learning Rate scheduler with a warm restart, decaying the learning rate from $2 \times 10^{-4}$ to $1 \times 10^{-7}$.
    

### 6.3 Competition Submission

For the final submission to the ICPR 2026 leaderboard:

1. **Ensembling:** We will train three instances of KSR-Net with different random seeds and slightly different backbones (Swin-Tiny, Swin-Small, Swin-Base). The final submission will be the majority vote of these three models.
    
2. **Formatting:** The output will be formatted strictly according to competition rules (CSV/JSON), likely requiring the License Plate text string and a confidence score.
    
3. **Submission Limits:** Respecting the limit of 5 submissions per day 1, we will use the validation set (a held-out portion of the training data) to rigorously vet models before uploading.
    

---

## 7. Conclusion

The **Kinematic-Structural Refinement Network (KSR-Net)** represents a comprehensive technical response to the ICPR 2026 LRLPR challenge. It systematically addresses the weaknesses of current SOTA methods by:

1. **Recovering lost data:** Using Hybrid Alignment (Flow + Deformable) to salvage information from frames that baseline methods would discard.
    
2. **Enhancing context:** Using 3D Swin Transformers to integrate temporal and spatial information more effectively than CNNs.
    
3. **Optimizing for the goal:** Using Task-Driven Semantic Loss to align the restoration process directly with the recognition objective.
    

By adhering to the forensic constraint of evidential value while maximizing the signal-to-noise ratio through intelligent aggregation, KSR-Net is positioned to define the new state-of-the-art in multi-frame license plate recognition.

---

References used in this report are integrated from the provided research snippets.3