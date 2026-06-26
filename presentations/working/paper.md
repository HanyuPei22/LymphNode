# DSN_2026_Hanyu_LymphNode

# LymphNode: A Plug-and-Play Access Control Method for Deep Neural Networks

Hanyu Pei Shang Liu Zeyan Liu Department of Computer Science Department of Computer Science Department of Computer Science and Engineering and Engineering and Engineering University of Louisville University of Louisville University of Louisville hanyu.pei@louisville.edu shang.liu@louisville.edu zeyan.liu@louisville.edu to decrypt the underlying static files. Despite these risks, ***Abstract *—Deep** **Neural** **Networks** **(DNNs)** **are** **high-value** **intel-** **lectual** **property** **(IP),** **yet** **deploying** **them** **to** **edge** **environments** on-premise deployment remains non-negotiable for sensitive **exposes** **them** **to** **unrestricted** **oracle** **access,** **rendering** **them** industries (e.g., healthcare, finance), creating a direct conflict **vulnerable** **to** **model** **extraction** **and** **inversion** **attacks.** **Existing** between operational necessity and IP security.

**defenses** **fail** **to** **address** **this** **practically:** **passive** **watermarking** Existing defense mechanisms struggle to resolve this con- **only** **offers** **post-hoc** **provenance,** **while** **active** **defenses** **impose** flict due to significant practicality limitations.*Passive defenses*, **prohibitive** **latency** **or** **require** **persistent** **access** **to** **sensitive** **training data. To bridge this gap, we propose *LymphNode *, a novel** such as watermarking [6]–[8], primarily serve copyright prove- **post-hoc** **defense** **framework** **that** **acts** **as** **an** **intrinsic** **“immune** nance *after* theft occurs, failing to proactively prevent func- **system”** **within** **the** **model.** ***LymphNode*** **enforces** **a** **strict** **“default-** tional extraction. Conversely, *active* *defenses* aim to enforce **deny” policy: it actively neutralizes model utility for unauthorized** access control but face critical bottlenecks. Cryptographic **queries** **via** **Generalized** **Sparse** **Universal** **Adversarial** **Perturba-** methods (e.g., Deep-Lock [9]) incur prohibitive computational **tions (GSUAP) injected into the feature space, effectively blocking** **gradient** **estimation** **and** **data** **inference.** **Utility** **is** **selectively** overhead, rendering them unsuitable for real-time edge ap- **restored** **only** **for** **authorized** **inputs** **carrying** **a** **stealthy** **feature-** plications. Meanwhile, structural authorization methods [10]– **domain** **credential.** **Our** **framework** **is** **highly** **practical:** **it** **is** [12] embed locking mechanisms but suffer from severe data **data-efficient,** **establishing** **robust** **protection** **with** **fewer** **than** **100** dependency: they typically require computationally expensive **samples** **(***<* 1% **of** **training** **data),** **and** **cross-dataset** **adaptable,** full model retraining or persistent access to the complete **enabling** **protection** **using** **public** **surrogate** **datasets.** ***LymphNode*** **thus** **provides** **a** **lightweight,** **immediately** **deployable** **defense** **for** original dataset. This makes them infeasible for real-world **high-stakes** **scenarios** **where** **original** **training** **data** **is** **restricted** scenarios where training data is sensitive, legally restricted **or** **unavailable.** (e.g., GDPR), or unavailable post-training. More recently, ***Index*** ***Terms *—Model** **IP** **Protection,** **Active** **Defense,** **Model** output perturbation methods [13], [14] actively degrade ex- **Extraction.** tracted model quality at the prediction interface, yet they require a runtime interposition layer inherent to cloud-API I. INTRODUCTION deployments and cannot protect models physically transferred Training modern machine learning models is an excep- to edge environments.

tionally resource-intensive endeavor. Large-scale foundation In this paper, we propose *LymphNode*, a post-hoc plugin models, such as GPT-4 or LLaMA [1], [2], require months framework designed to bridge these practicality gaps. Unlike of computation, making their weights invaluable intellectual invasive methods requiring retraining, our approach integrates property (IP). However, to satisfy latency and privacy require- a lightweight “checkpoint” directly into the inference pipeline.

ments, models are frequently deployed to edge devices or on- Drawing inspiration from the biological immune system,*Lym-* premise servers. While this decentralization protects raw data, *phNode* enforces a rigorous “default-deny” policy. It treats it inadvertently grants adversaries unrestricted oracle access all inputs as unauthorized and injects a pre-computed, Gen- to the model interface. Unlike rate-limited cloud APIs, edge eralized Sparse Universal Adversarial Perturbation (GSUAP, deployment allows attackers to query the model with infinite detailed in Sec. III-C) into critical channels. This defense volume at zero latency. Exploiting this, adversaries can launch persistently neutralizes the model’s output quality for arbitrary sophisticated model extraction attacks [3], [4] to reverse- queries, thereby significantly degrading the quality of outputs engineer parameters or model inversion attacks [5] to infer available for model extraction [3], [15] and inversion [5], [16] sensitive training data. This capability allows malicious actors attacks. Access is restored only when an authorized input with to functionally replicate proprietary models without needing a specific feature-domain credential is verified, triggering an inverse perturbation to cancel the noise. Consequently, the © 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, model remains useless to adversaries relying on oracle access, including reprinting/republishing this material for advertising or promotional while authorized users transparently recover full fidelity. The purposes, creating new collective works, for resale or redistribution to servers overall architecture is illustrated in Fig 1. In summary, our key or lists, or reuse of any copyrighted component of this work in other works.

<!-- Page 2 -->

contributions are:

TABLE I QUALITATIVE COMPARISON WITH STATE-OF-THE-ART ACTIVE IP *•* We propose *LymphNode*, a novel post-hoc plugin frame- PROTECTION METHODS.*LymphNode* UNIQUELY COMBINES DATA work that provides active IP protection. It actively neu- INDEPENDENCE, ZERO-RETRAINING, AND NEGLIGIBLE OVERHEAD.

tralizes models against unauthorized input via a sparse adversarial perturbation, while restoring full fidelity for **Method** **Origin** **Data** **Retrain** **Post-Hoc** **Overhead** authorized users through a stealthy, feature-domain veri- ✗ ✗ ✓ Deep-Lock [9] High fication protocol.

✓ ✓ ✗ AdvParams [12] Low *•* Our framework demonstrates exceptional**data adaptivity** ✓ ✓ ✗ SSAT [11] Low **and efficiency**. We prove that the framework significantly ✓ ✓ ✗ IDEA [29] Medium lowers the deployment barrier: it can be robustly ini- ✓ ✓ ✗ ModelLock [30] Low tialized using as few as 50-100 samples (*<* 1% of the ✗ ✗ ✓ **Ours** **Negligible** training set) and is capable of protecting target models using public surrogate datasets (e.g., protecting STL-10 models with CIFAR-10 noise). This eliminates the strict copyright provenance which serves to prove ownership after dependency on original private data required by prior art.

a theft has already occurred, typically for litigation purposes.

*•* We conduct a comprehensive security and robustness As comprehensively analyzed by Lukas et al. [23], while such evaluation. We rigorously validate the system’s resilience methods can verify ownership with high confidence, they pos- against adaptive threats, including deep generative cre- sess no capability to proactively prevent unauthorized model dential forgery and model hijacking via fine-tuning. Fur- execution. Consequently, once a protected model is leaked or thermore, we demonstrate that the protection mechanism distributed, adversaries can freely deploy and monetize the remains dependable under real-world distortions (e.g., asset, leaving the window of exploitation entirely open despite lossy compression), while maintaining near-perfect wa- the presence of ownership proofs.

termark imperceptibility (LPIPS *≈*0*.*001), confirming its suitability for practical deployment.

*B.* *Recent* *Advances* *in* *UAP* The remainder of this paper is organized as follows. Sec- Recent scholarship in adversarial learning has produced tion II reviews related work in model watermarking and active highly effective strategies for generating universal adversarial defenses. Section III details the architecture and the post- perturbations (UAP). Techniques such as SGA [24], DM- hoc training mechanism of *LymphNode*. Section IV presents UAP [25], and RobustUAP [26] introduce sophisticated opti- the experimental setup and a comprehensive evaluation of mization objectives, including stochastic gradient aggregation effectiveness, efficiency, and robustness. We further discuss and dynamic maximin frameworks, to maximize the trans- the data adaptivity and robustness of our framework in Sec. V ferability and robustness of attacks across diverse models and Sec. VI. Finally, we conclude the paper in Section VIII.

and distributions. Parallel research like Sparse-PGD [27] has further refined the generation of sparse adversarial noise.

II. RELATEDWORK However, these methods are fundamentally engineered for *A.* *Passive* *Defense:* *IP* *Provenance* offensive generalization rather than defensive controllability.

Passive defense mechanisms for DNN intellectual prop- Their primary goal is to construct an irreversible perturba- erty protection predominantly consist of model watermark- tion that degrades performance across unknown target do- ing and fingerprinting techniques. Existing watermarking ap- mains, a process that typically necessitates extensive train- proaches [6], [7], [17] embed identifying information directly ing data to approximate universal vulnerability manifolds. In into model parameters [7], activation maps [17], or couple the context of active IP protection via a lightweight plu- them with backdoor triggers [6], [18], [19] to enable owner- gin framework, the operational objective is fundamentally ship verification. In parallel, fingerprinting methods [20], [21] different. We require a targeted neutralization mechanism extract intrinsic model attributes, such as decision boundary capable of locking a specific, known model with minimal characteristics or specific prediction behaviors on adversar- setup costs. Consequently, our work adopts the Generalized ial examples, to construct unique identifiers without altering Data-Free UAP (GD-UAP) [28] formulation. By leveraging the original model weights. Collectively, these embedded or its strong data-independent characteristics, we can efficiently extracted signatures allow model owners to verify authorship optimize the perturbation to saturate the target model’s feature through statistical analysis or behavioral querying. More re- space without the prohibitive data dependency associated with cently, DynaMarks [22] embeds transferable watermarks into transferability-oriented attacks.

extracted surrogates by dynamically altering output proba- *C.* *Active* *Defense:* *Cryptographic* *Approaches* bilities at inference time, yet the extracted model retains full functionality and the watermark serves only for post-hoc An alternative paradigm involves cryptographic tech- ownership claims.

niques that provide active protection during inference.

Hardware-based solutions like Trusted Execution Environ- However, these approaches fundamentally constitute a pas- ments (TEEs) [31] and algorithmic solutions like Homo- sive line of defense. Their primary utility is restricted to

<!-- Page 3 -->

|Clean Image<br>(𝒙)|Watermark<br>(𝒘)|
|---|---|

**LymphNode Plugin** **High-Fidelity** **Output**

|Col1|Col2|
|---|---|
|Feature-level<br>Authorization key<br>𝒚𝒌||

Antidote **Probability** **Reverse** **Engineering** 𝒚𝒌 **Yes** LymphNode Checkpoint **8** Is𝒚𝒌 Correct Matched?

**Processed** Classification BN1 Inverse Noise **Features** **No** (-ℳ⨀Δ) **Authorized** Layer Conv1 **Input**(𝒙𝒘) & Layer Rest of ℱ(∙) **Utility** DNN **Neutralized** **Feature map** Sparse Noise Mask(ℳ) GSUAP Injection **Probability** **Unauthorized** **Perturbed** **Input** **Features** **Frozen Target DNN** Random Guess Masked Noise(ℳ⨀Δ) **(Backbone)** Fig. 1. An overview of LymphNode plugin *E.* *Active* *Defense:* *Anti-Extraction* *via* *Output* *Perturbation* morphic Encryption (HE) [32] theoretically guarantee secure execution. Similarly, methods like Deep-Lock [9] and NN- A parallel line of work actively degrades extracted models Lock [33] propose S-Box-based parameter encryption, requir- by perturbing responses at the prediction interface. Methods ing decryption for every query. However, recent surveys [34] like Prediction Poisoning [13], CIP [14], and AMAO [36] highlight significant practical barriers: TEEs are susceptible dynamically inspect, score, or poison per-query outputs to to side-channel attacks, while HE and parameter decryption disrupt attacker objectives.

schemes impose prohibitive computational overhead and la- These methods strictly require a server-side interposition tency penalties. This renders them impractical for resource- layer—a capability inherent to cloud-APIs but physically constrained edge deployments where real-time inference is infeasible at the edge, where adversaries directly invoke the non-negotiable.

forward pass. In contrast, *LymphNode* embeds protection statically within the computational graph as a feature-space *D.* *Active* *Defense:* *Structural* *Authorization* intervention, eliminating runtime query analysis.

More recent works aim to achieve active authorization con- III. FRAMEWORK trol by modifying the model structure or weights. Represen- tative methods include ModelGuard [10], ActiveGuard [35], In this section, we present *LymphNode*, a post-hoc defense SSAT [11], and AdvParams [12]. These approaches embed framework that structurally integrates active IP protection backdoors or adversarial perturbations into the weights during directly into the target Deep Neural Network (DNN). Unlike training to discriminate between authorized and unauthorized external pre-processing wrappers, our approach fuses the se- users. Recent advancements like ModelLock [30] leverage curity mechanism into the model’s computational graph as an diffusion models to edit the training distribution for enhanced intrinsic intervention node, specifically targeting the interme- locking.

diate feature space. We operate within a standard black-box Crucially, these approaches suffer from a severe **setup** **cost** deployment scenario where the target model is encapsulated **bottleneck**. They typically require either: (i) computationally in an inference environment that preserves runtime integrity.

expensive full model retraining (e.g., ModelLock, SSAT), The framework governs the inference logic by enforcing a strict “default-deny” policy embedded within the forward or (ii) persistent access to the complete original training pass topology: a *Generalized* *Sparse* *Universal* *Adversarial* dataset (e.g., AdvParams, IDEA [29]) to compute gradients or *Perturbation* (GSUAP, detailed in Sec. III-C) is injected into generate triggers. This requirement is often infeasible in real- the latent features by default, neutralizing model utility for world scenarios due to strict privacy regulations (e.g., GDPR) any unauthorized access. Full fidelity is only recovered when or data loss. Even sample-specific approaches remain bound to the original training pipeline. To clearly distinguish our a coupled antidote mechanism validates a stealthy feature- domain credential, thereby rendering the security logic insepa- contribution within the active defense landscape, we provide rable from the model’s fundamental feature extraction process.

a qualitative comparison in Table I.

<!-- Page 4 -->

receptive fields corresponding to the *N* features and then em- *A.* *Threat* *Model* *and* *Assumptions* ploy a random search to identify a valid**w**. The computational A three-party edge deployment scenario is considered. The complexity is *O*(*h**×*2*v*), which is highly efficient given the *Model Owner*trains the target DNN, optimizes the*LymphNode* bound *N* = *v* *×* *h*. Empirical results in Sec. IV-C confirm GSUAP plugin, and issues authorization keys to the *Edge* that generating 1,000 unique credentials requires less than 2 *Operator* via a secure out-of-band channel. The *Edge* *Op-* seconds.

*erator* deploys the protected model within a trusted runtime Finally, the verification mechanism necessitates a funda- environment (e.g., signed firmware) that preserves parameter mental security trade-off. The choice of *N* balances effi- integrity but exposes model functionality through an inference ciency and specificity. A smaller *N* accelerates generation API. Crucially, the*Edge Operator*independently manages key but increases the risk of an unauthorized input coincidentally distribution to*Authorized End-Users*within their trust domain.

matching **k** (a “collision”), which would grant illicit access.

The adversary model is strictly gray-box: the adversary The theoretical collision probability*P**c*is estimated as follows:

possesses knowledge of the model architecture and the gen- eral defense mechanism, but lacks access to (i) the secret *N* Y *P*(*b**i* =*k**i*) = 2*−**N* *P**c* =*P*(*b*1 =*k*1*, . . . , b**N* =*k**N*) = authorization key **k**, (ii) runtime memory or intermediate feature maps, and (iii) gradient or backward-pass computa- *i*=1 (3) tions. Oracle access is therefore the only available attack where *P*(*b**i* = *k**i*) denotes the collision probability for the surface: the adversary may submit arbitrary queries—including *i**th* bit. The Eq. 3 holds when *P*(*k**i* = 0) = *P*(*k**i* = 1) = adaptive queries crafted from previous outputs—and observe 0*.*5 and *P*(*k**i*) = *P*(*k**i**|**k**j*)*,**∀**i* *̸*= *j*, indicating the bit-wise the returned probability vectors.

collision follows binomial distribution. These conditions are verified to hold by experiments in [37]. For our setting (*N* = *B.* *Authorization* *via* *Feature-Domain* *Verification*

32. , *P**c* *≈*2*.*33*×*10*−*10. This infinitesimal probability ensures

As the core decision logic of the *LymphNode* framework that the “default-deny” policy is robustly enforced, preventing (Fig. 1), this module implements identity verification by em- accidental authorization by benign inputs.

bedding a discrete credential into the continuous feature space.

The authorization key is defined as a secret*N*-bit binary string *C.* *Model* *Performance* *Neutralization* *via* *GSUAP* **k***∈{*0*,*1*}**N*. To embed this key, we select *N* carrier features To efficiently regulate the model’s fidelity for unauthorized from the first convolutional layer’s output using a distributed access, we propose **Generalized** **Sparse** **Universal** **Adver-** strategy and denote the feature representation of authorization **sarial** **Perturbations** **(GSUAP)**. While standard Universal key as **y***k*. Assuming the layer comprises *r* kernels with a Adversarial Perturbations (UAPs) [28], [38] seek to fool a spatial size of *m* *×* *n*, we select *v* distinct kernels and *h* model on all inputs, applying them indiscriminately increases distinct spatial locations within each kernel, such that the total computational overhead and detection risk. Therefore, our capacity is *N* =*v**×**h* (noticing that these *h* features from the objective is to adapt structured pruning principles [39], [40] to same feature map correspond to non-overlapping pixel-domain identify a minimal subset of *decision-critical* *channels*, where regions). Based on the ablation study in prior work [37], we targeted noise injection can maximally disrupt model behavior adopt a configuration of*N* = 32and*v* = 4to balance capacity while minimizing the modification footprint.

and stealthiness.

Performance Neutralization operates in two sequential For each carrier feature *y*, we target the *s*-th bit after the phases: first, channels are selected based on their gradient binary point for verification. Mathematically, the extraction of sensitivity regarding the classification loss; second, a constant the verification bit *b* is formulated as:

adversarial perturbation is optimized on these selected chan- *b*=*⌊|**y**| ·*2*s**⌋*mod 2 nels.

(1) **Phase** **1:** **Weight** **Gradient-based** **Channel** **Selection.** Intuitively, a larger *s* implies that the verification relies on Given a pre-trained clean model *f**θ* and a small calibration finer-grained quantization noise, resulting in smaller perturba- dataset *D**cal*, we quantify each channel’s importance. Draw- tions and better stealthiness. Following Liu et al. [37], we set ing from gradient-based pruning criteria [39], we define the *s* = 6, rendering the modifications visually imperceptible in importance score for the *j*-th channel in a layer (with weight the pixel domain.

tensor *W*) as the expected gradient magnitude:

To pass this verification, an authorized user must generate *∂**L**CE*(*f**θ*(*x*)*, y*) an input **x***auth* = **x**+**w** such that its feature representation Score*j* =E(*x,y*)*∼D**cal* (4) *∂W**j* matches **k**. This constitutes an inverse problem: finding a 2 perturbation **w** that satisfies:

where *W**j* denotes the kernel weights for channel *j*. A high score indicates that the loss is highly sensitive to variations *F*(**x**+**w**) =**y k** (2) in this channel. Given a target sparsity ratio *r* *∈*(0*,*1], we select the top-*k* (*k* =*⌊**r**·**M**⌋*) channels to construct a binary where *F* represents the first convolutional layer. Since direct mask *M* *∈{*0*,*1*}**M*. Unlike magnitude-based ranking [41] analytical inversion is ill-posed due to dimensionality mis- match, we employ a search-based strategy. We first locate the which evaluates static weights, this gradient-based criterion

<!-- Page 5 -->

**Sparse UAP (SUAP)**. SUAP is formulated by applying a spar- dynamically identifies the optimal “acupuncture points” for sity mask to a standard UAP: ∆*sparse* =*M⊙*∆. To ensure a performance neutralization.

**Phase** **2:** **Sparse** **Adversarial** **Noise** **Optimization.** With fair comparison, all three strategies utilize the identical sparsity mask *M* generated by our Weight-Gradient selector, the channel mask *M* fixed and model parameters frozen, we optimize a universal additive noise ∆to maximize misclassi- and their perturbation magnitudes are constrained to the same budget (*∥**ϵ**∥**∞*= 2, applies to the normalized feature space fication on unauthorized inputs. For any input *x*, the noise is after Batch Normalization). We integrate *LymphNode* into a injected into the selected channels of the feature map:

diverse suite of architectures to verify broad applicability, ˜*F*(*x*) =*F*(*x*) +*M ⊙*∆ (5) including standard residual networks (ResNet-18, ResNet- 50 [45]), vision transformers (ViT-Tiny, ViT-Small [46]), and where *F*(*x*) is the clean feature map and *⊙*denotes element- classic architectures (AlexNet [47], DenseNet [48]). For each wise multiplication. To find the optimal perturbation, we dataset, we randomly sample 2000 images, add watermark to employ Projected Gradient Ascent (PGA) to maximize the classification loss *L**CE*. Let *g*(*t*) denote the gradient of 1000 of them as authorized input, while the rest as unautho- the loss with respect to the noise at step *t*, i.e., *g*(*t*) = rized input. We quantify effectiveness via the *Unauthorized* *∇*∆*L**CE*(*f*(*x*; ∆(*t*))*, y*). The update rule is formulated as:

*Accuracy* (lower is better) to measure the suppression effect, and the *VIP* *Accuracy* (higher is better) to demonstrate the ∆(*t*+1) = Π*ϵ* (∆(*t*) +*α**·*sign(*g*(*t*)))*⊙M* (6) extent of performance preservation. The quantitative results are presented in Table II.

where *α* is the step size, Π*ϵ* projects the noise onto the The quantitative results demonstrate that GSUAP univer- *ℓ**∞*-ball bounded by *ϵ*. Crucially, this optimization targets sally outperforms baselines, evidenced by its ability to sup- the unauthorized scenario: the goal is to find a single, static press ResNet-18 accuracy on CIFAR-10 to 13.6% at a 60% perturbation ∆that, when masked by *M*, causes the model ratio, whereas Gaussian noise fails (85.4%). This disparity to fail on clean inputs. Once optimized, this noise module confirms that structural corruption alone is insufficient, high- is integrated into the *LymphNode* plugin as shown in Fig. 1.

lighting the necessity of gradient-guided semantic destruction It remains active by default to neutralize performance for for neutralizing robust models. We further observe that simpler unauthorized users, and is only bypassed (via the antidote datasets and densely connected architectures like DenseNet mechanism, an inverse GSUAP) when a valid feature-domain exhibit inherent fragility to perturbations due to error propa- credential is verified.

gation, whereas the resilience of ResNet models validates the IV. EXPERIMENTATION requirement for targeted adversarial interference. Throughout these configurations, authorized inputs maintain optimal accu- In this section, we establish the foundational performance of racy, ensuring no performance penalty for legitimate users.

*LymphNode*, focusing on its core capability to regulate model These observations validate the effectiveness of the *Lym-* inference under realistic deployment constraints. We rigor- *phNode* framework. The consistent superiority of GSUAP ously evaluate the framework along three primary dimensions, establishes a reliable protection boundary that stochastic base- beginning with the neutralization effectiveness, where we lines cannot achieve, particularly on robust architectures. Si- verify the system’s ability to selectively suppress unauthorized multaneously, the preservation of authorized fidelity confirms accuracy across diverse architectures and datasets. Concur- the precision of our feature-domain verification mechanism, rently, we assess the neutralization efficiency to quantify the which effectively decouples authorized flows from the noise protection gain relative to the structural modification cost. To injection path. In summary, *LymphNode* provides a robust ensure practical viability, we also analyze system performance paradigm for active model protection, reconciling strict unau- by benchmarking computational overheads, including latency, thorized lockout with seamless authorized access.

throughput, and memory, validating the framework’s suitability for resource-constrained environments. Comprehensive analy- *B.* *Neutralization* *Efficiency* *Analysis* ses regarding design choices, data dependencies, and security While the absolute performance drop is a critical measure resilience are presented in subsequent sections.

of security, a practical protection mechanism must also be *A.* *Neutralization* *Effectiveness* *Evaluation* efficient, achieving maximum security with minimal structural modification. To rigorously quantify the return on investment In this section, we evaluate the capability of *LymphNode* for each injected noise channel, we introduce the Neutraliza- to selectively neutralize model performance for unauthorized tion Efficiency metric*E*. Formally, defined by the performance access. As detailed in Sec. III-C, our framework injects Gener- gap between authorized and unauthorized users normalized alized Sparse Universal Adversarial Perturbations (GSUAP) to by the sparsity ratio of modified channels *ρ*, the metric is suppress inference accuracy when valid credentials are absent.

calculated as:

To rigorously benchmark the neutralizing potency of *E* = *A**auth**−A**unauth* GSUAP, we conduct a comprehensive evaluation on three (7) *ρ* benchmark datasets: CIFAR-10 [42], MNIST [43], and where*A**auth*and*A**unauth*represent the classification accuracy SVHN [44]. We construct two comparative baselines by adapt- ing alternative neutralizing strategies: **Gaussian** **Noise** and for authorized and unauthorized (Normal) inputs, respectively.

<!-- Page 6 -->

TABLE II NEUTRALIZATION EFFECTIVENESS ACROSS DATASETS AND ARCHITECTURES.*†* **CIFAR-10** **MNIST** **SVHN** **Model** **Ratio** **(%)**

|Gauss SUAP GSUAP VIP|Gauss SUAP GSUAP VIP|
|---|---|

Gauss SUAP GSUAP VIP Gauss SUAP GSUAP VIP Gauss SUAP GSUAP VIP

|20<br>40<br>ResNet-18 60<br>80<br>100|93.6 93.2 88.6 94.5<br>87.4 81.4 36.4 94.5<br>85.4 72.0 13.6 94.5<br>80.2 67.0 11.0 94.5<br>73.6 48.4 10.6 94.5|49.0 10.2 10.2 99.6<br>31.3 11.3 11.3 99.6<br>29.4 9.3 9.3 99.6<br>24.7 10.6 10.2 99.6<br>22.6 10.2 10.2 99.6|64.6 13.2 8.1 96.1<br>60.3 11.1 7.4 96.1<br>50.8 10.9 7.2 96.1<br>46.5 10.8 6.6 96.1<br>44.0 10.8 6.6 96.1|
|---|---|---|---|

|20<br>40<br>ResNet-50 60<br>80<br>100|95.2 87.2 79.2 95.8<br>83.8 65.2 47.8 95.8<br>74.2 47.8 25.6 95.8<br>72.6 42.0 11.2 95.8<br>68.6 32.4 9.0 95.8|92.6 16.7 16.3 99.8<br>45.7 11.1 10.9 99.8<br>35.3 10.4 10.2 99.8<br>33.7 10.4 10.2 99.8<br>32.0 10.4 10.0 99.8|68.4 25.3 10.8 96.6<br>65.8 21.4 9.5 96.6<br>59.4 19.4 9.8 96.6<br>52.7 19.2 9.5 96.6<br>48.6 19.2 9.4 96.6|
|---|---|---|---|

|20<br>40<br>ViT-Tiny 60<br>80<br>100|71.6 49.0 27.8 91.8<br>51.0 16.6 11.6 91.8<br>39.8 13.6 11.8 91.8<br>34.8 11.8 10.6 91.8<br>29.8 11.2 10.6 91.8|46.3 11.2 10.4 99.3<br>32.4 10.9 10.0 99.3<br>23.8 9.4 8.9 99.3<br>21.2 9.9 8.4 99.3<br>19.0 9.8 8.4 99.3|67.8 19.8 24.1 96.3<br>43.0 18.4 18.1 96.3<br>32.5 11.2 10.8 96.3<br>26.0 10.3 10.8 96.3<br>22.2 10.2 10.8 96.3|
|---|---|---|---|

|20<br>40<br>ViT-Small 60<br>80<br>100|55.6 22.6 13.6 89.4<br>36.0 15.2 10.2 89.4<br>27.0 11.2 10.0 89.4<br>24.8 10.6 10.0 89.4<br>23.6 11.6 10.0 89.4|40.8 10.8 10.7 99.2<br>26.1 10.0 10.0 99.2<br>22.7 9.8 9.4 99.2<br>17.9 9.5 9.5 99.2<br>14.1 9.5 9.4 99.2|43.7 23.6 21.8 97.6<br>25.9 22.0 21.0 97.6<br>20.0 18.2 18.9 97.6<br>20.0 13.2 14.6 97.6<br>20.0 13.2 11.0 97.6|
|---|---|---|---|

|20<br>40<br>DenseNet 60<br>80<br>100|14.2 13.5 11.3 95.8<br>10.7 10.7 9.6 95.8<br>10.6 9.6 9.3 95.8<br>10.2 9.7 9.0 95.8<br>10.2 9.5 9.0 95.8|12.3 10.7 9.5 99.5<br>10.1 9.8 9.8 99.5<br>10.3 10.3 9.3 99.5<br>10.5 9.5 9.5 99.5<br>10.4 9.4 9.4 99.5|15.8 15.6 9.9 96.1<br>15.4 14.6 8.0 96.1<br>13.1 14.1 7.7 96.1<br>10.5 9.4 7.6 96.1<br>9.9 8.5 7.7 96.1|
|---|---|---|---|

**11.0** **9.7** **10.5** 20

23. 5
12. 7
90. 5
12. 6
9. 8
98. 9
28. 0
15. 7
95. 4

**11.5** **9.4** **7.3**

20. 8
11. 9
90. 5
11. 5
9. 5
98. 9
27. 3
11. 7
95. 4

**9.8** **9.0** **6.0**

12. 7
9. 9
90. 5
10. 9
9. 2
98. 9
26. 8
8. 0
95. 4

AlexNet **9.0** **8.9** **6.0**

11. 0
9. 2
90. 5
10. 6
9. 2
98. 9
20. 7
7. 1
95. 4

**9.0** **8.9** **6.0**

11. 4
9. 2
90. 5
10. 6
9. 2
98. 9
16. 7
7. 1
95. 4

*†***Ratio**: percentage of channels with GSUAP injection. Gauss/SUAP/GSUAP: accuracy of unauthorized inputs (%, lower is better). VIP: accuracy of authorized inputs (%, higher is better). **Bold**: best suppression; underline: second best.

ResNet-18 ResNet-50 DenseNet

2. 5
4. 0
2. 5
3. 5
2. 0
2. 0

Efficiency Score

3. 0
1. 5
1. 5
2. 5
2. 0
1. 0
1. 0
1. 5
0. 5
0. 5
1. 0

20 20 20 AlexNet ViT-Tiny ViT-Small

4. 0
4. 0
3. 5
3. 5
3. 5
3. 0

Efficiency Score

3. 0
3. 0
2. 5
2. 5
2. 5
2. 0
2. 0
2. 0
1. 5
1. 5
1. 5
1. 0
1. 0
1. 0

20 20 20 Channel Ratio (%) Channel Ratio (%) Channel Ratio (%) Gauss SUAP GSUAP Fig. 2. Neutralization Efficiency.

Mathematically, this represents the marginal neutralization utility, quantifying how much protection gain is achieved per

<!-- Page 7 -->

unit of channel resource consumed. To provide a holistic FLOPs (Computational Complexity), inference latency (Batch Size=1), throughput (Batch Size=128), and peak memory view of architectural behaviors that is not biased by specific usage. We select ResNet-18 and ViT-Tiny as representative data characteristics, we compute the efficiency scores for architectures for CNNs and Transformers, respectively. Test each model by averaging the results across all three datasets data is sampled from watermarked CIFAR-10 dataset.

(CIFAR-10, MNIST, and SVHN).

As illustrated in Figure 2, the efficiency trajectories across six architectures reveal distinct behavioral patterns governed TABLE III by both perturbation strategy and model topology. Universally, SYSTEM OVERHEAD ANALYSIS.

the efficiency curves exhibit a monotonic decay as the channel **ResNet-18** **ViT-Tiny** ratio increases, a phenomenon consistent with the law of **Metric** diminishing returns where the initial set of critical channels **Cost** **Cost** Clean Prot.

Clean Prot.

contributes most significantly to the decision boundary. Cru-

|Params (M)<br>FLOPs (G)<br>Latency (ms)<br>Throughput (/s)<br>Memory (MB)|11.17 11.17 +0.0%<br>0.558 0.558 ≈0<br>1.70 2.74 +1.0<br>6893 5875 -14.8%<br>93.2 109.2 +17.2%|1.80 1.80 +0.0%<br>0.118 0.118 ≈0<br>1.54 2.60 +1.1<br>17175 16023 -6.7%<br>33.7 35.2 +4.5%|
|---|---|---|

cially, GSUAP consistently establishes a superior efficiency frontier compared to Gaussian noise and SUAP, particularly on robust architectures like ResNet and ViT. For instance, at a sparsity ratio of 20% on ResNet-18, GSUAP achieves an efficiency score of nearly 2.5, significantly outperforming Gaussian noise (*<* 0*.*8). This empirical evidence confirms **Results** **Analysis.** Table III summarizes the results. First, that GSUAP does not merely introduce stochastic noise but regarding theoretical complexity, the plugin introduces negli- precisely targets the model’s semantic vulnerabilities. We gible overhead in terms of parameters and FLOPs (*<*0*.*01%).

further observe a divergence based on architectural resilience:

This confirms that the element-wise operations (LSB ex- for densely connected structures like DenseNet and legacy traction and noise injection) are computationally lightweight models like AlexNet, the efficiency curves of all three methods compared to the backbone’s convolutional layers. Second, nearly collapse into a single trajectory, corroborating the “error regarding runtime performance, the throughput reduction is avalanche” effect where inherent architectural fragility renders modest (*−*6*.*7% to *−*14*.*8%), maintaining high efficiency for the sophistication of the noise generation secondary to the batch processing. Finally, for real-time latency, although the perturbation itself. Conversely, for modern architectures pos- relative increase appears significant due to the extremely low sessing structural redundancy, such as ResNets and ViTs, the baseline of CIFAR models, the absolute latency penalty is performance gap between GSUAP and the baselines widens consistently around 1.0 ms. In practical scenarios, such as significantly.

video processing at 30 FPS (requiring *<*33 ms per frame), a Synthesizing these observations, we conclude that the effi- total inference time of*≈*2*.*7ms is well within the operational ciency of gradient-guided adversarial perturbations is funda- envelope. This confirms that *LymphNode* achieves strong ac- mentally superior to stochastic interference, particularly when tive protection without compromising system responsiveness.

protecting robust models. The pronounced “efficiency gap” observed on resilient architectures (e.g., ResNet) highlights *D.* *Channel* *Selection* *Evaluation* that while random noise may suffice for fragile models, the To validate the channel selection strategy within *Lym-* precise, adversarial nature of GSUAP is indispensable for *phNode*, we conduct an ablation study on the CIFAR-10 neutralizing robust networks under strict resource constraints.

dataset to determine the most effective criterion for identifying Consequently, these results validate the core design premise of critical features. We benchmark our **Weight** **Gradient** ap- *LymphNode*: by leveraging GSUAP, our framework achieves proach against three established saliency metrics derived from substantial model degradation with minimal feature modi- model pruning literature: Random selection (baseline),**Weight** fication. This high operational efficiency ensures that the **Norm** [40], and **Taylor** **Expansion** [49]. To ensure a strictly protection mechanism imposes the lowest possible structural fair comparison, we apply these criteria to generate the sparsity cost while maintaining a robust lockout against unauthorized mask while keeping the noise injection method (GSUAP) access.

constant. The evaluation focuses on four representative ar- *C.* *System* *Performance* *Analysis* chitectures: ResNet-18, ResNet-50, ViT-Tiny, and ViT-Small.

For any active protection mechanism to be viable in real- These models are selected because their inherent structural world deployments—particularly on resource-constrained edge redundancy offers high resilience to interference, providing a devices—it must impose minimal computational burden. In rigorous testing ground to distinguish the efficacy of different this section, we rigorously quantify the system overhead selection strategies. All necessary selection metrics, such as introduced by the *LymphNode* plugin.

gradients and weight magnitudes, are computed directly from the corresponding clean models pre-trained on CIFAR-10. We In detail, we benchmark the inference performance on quantify the performance using the Accuracy Gap (defined an NVIDIA GeForce RTX 4060 Laptop GPU. We compare as *A**auth* *−A**unauth*), where a wider gap indicates a more the original clean models against their protected counterparts efficient selection method that achieves greater performance across five key metrics: parameter count (Storage), theoretical

<!-- Page 8 -->

ResNet-18 ResNet-50 ViT-Tiny ViT-Small Accuracy Gap (%) 20 0 20 20 20 20 Channel Ratio (%) Channel Ratio (%) Channel Ratio (%) Channel Ratio (%) Random (Baseline) Weight Norm Taylor Weight Gradient (Ours) Fig. 3. Ablation study for selector.

degradation for unauthorized users while maintaining the same sparse footprint. The results are presented in Figure 3.

The results provide a clear justification for our design choice. First, we observe that perturbing a specific proportion of critical channels (e.g., 60% on ResNet and 40% on ViT) yields neutralization effects comparable to full-channel per- turbation, verifying that targeted noise injection significantly enhances efficiency without compromising potency. Univer- sally, the selection strategy proves decisive; both gradient- based methods significantly outperform the magnitude-based and random baselines, confirming that sensitivity to the loss Impact of Noise Scale (*λ*) on model performance (ResNet-18 on Fig. 4.

function is a superior proxy for channel importance compared CIFAR-10).

to static magnitude heuristics. Crucially, between the top per- formers, our **Weight** **Gradient** method demonstrates superior stability and overall efficiency. While **Taylor** **Expansion** is As illustrated in Figure 4, the performance trajectories competitive at extreme sparsity levels for smaller models, reveal a contrast between user groups. The authorized User it lacks consistency in the practical 40%–80% range. For accuracy (blue line) remains robustly stable at the optimal instance, on ResNet-50 at a 60% ratio, our method achieves an level (*≈*99*.*8%) across the entire range of *λ*, confirming that Accuracy Gap of approximately 75%, substantially surpassing our identity verification module reliably triggers the antidote the 52% achieved by Taylor Expansion. Furthermore, on ViT- mechanism. Conversely, the performance for unauthorized Tiny, our approach maintains a distinct performance lead users (red line) exhibits a smooth, monotonic decline as *λ* across all ratios. Consequently, we adopt the weight gradient increases. This experiment validates the controllability of our criterion as the default selector for our framework due to its framework, demonstrating that *LymphNode* enables flexible robust trade-off between sparsity and neutralization capability.

deployment strategies ranging from subtle quality degradation to complete functionality lockout without the need to retrain *E.* *Impact* *of* *Noise* *Scale* the model.

A distinct advantage of our plugin-based architecture is the *F.* *Real* *Attack* *Scenarios* ability to dynamically regulate the severity of performance degradation. Unlike weight-embedded backdoors that typically To directly validate the defense against practical IP offer a binary “all-or-nothing” outcome, *LymphNode* allows threats, we evaluate *LymphNode* under two representative the model provider to tune the intensity of the injected Gener- attack pipelines with oracle-access conditions consistent with alized Sparse Universal Adversarial Perturbation (GSUAP) to Sec. III-A. We include PP [13] and CIP [14] as baselines.

Since both methods are originally designed for cloud-API achieve granular access control. To demonstrate this capability, we conduct a representative case study using ResNet-18 on deployment with runtime query interposition, we preserve the CIFAR-10 dataset. We fix the sparse channel selection their core defense logic (per-query gradient perturbation for ratio at 40% and systematically vary the noise scale factor PP, reliability-based poisoning for CIP), remove the query- *λ* from 0*.*0 to 2*.*0 with a step size of 0*.*1. Here, *λ* = 0 detection components that require server-side control, and represents an unprotected model, while higher values indicate activate their defense on all queries unconditionally. This stronger interference. We measure the inference accuracy for adaptation is strictly favorable to the baselines, as it eliminates both authorized users and unauthorized users.

any detection failure and ensures 100% defense activation.

<!-- Page 9 -->

*Model* *Extraction.* We adopt the KnockoffNets [15] protocol, = 4.17%, Acc-5 = 11.22%, both near random guessing) with where an adversary trains a substitute model from input–output only 7.3% latency increase (0.088 ms), as GSUAP is a pre- pairs collected from the victim. The target is ResNet-18 on computed static tensor requiring only a constant-cost element- CIFAR-10 with channel ratio 60% and noise scale 2.0. The wise addition. For throughput-sensitive edge applications such query budget varies from 1,000 to 50,000; results are reported as real-time video analytics, this constant-overhead property in Table IV.

is essential—any per-query computation directly erodes the available compute budget on resource-constrained devices.

TABLE IV Across both attacks, *LymphNode* matches or exceeds the KNOCKOFFNET EXTRACTION ATTACK ON CIFAR-10.

strongest baseline in protection while adding negligible infer- ence cost, making it uniquely suited for latency-constrained **Budget** **No** **Def.** **PP** **[13]** **CIP** **[14]** **Ours** edge deployment.

Victim Acc.

92. 54
92. 54
92. 54
92. 54

V. DATAADAPTIVITY **13.71** 1,000

42. 88
25. 26
28. 44

In this section, we use two experiments to highlight that **14.76** 5,000

56. 25
32. 68
35. 17

LymphNode has strong dataset adaptivity. We validate that **14.68** 10,000

66. 16
38. 24
42. 81

our plugin framework can achieve performance control with **15.28** 50,000

85. 24
45. 66
48. 25

limited data sampling from a similar domain.

Ours: non-VIP model accuracy = 16.47%, output entropy = 1.35 (uniform = *A.* *Dataset* *Size*

2. 30).

Here, we evaluate the influence of calibration data size on Without defense, surrogate accuracy reaches 85.24% at 50K the effectiveness of *LymphNode*. Using the ResNet-18 archi- queries. PP and CIP reduce this to 45.66% and 48.25%, yet tecture on CIFAR-10 as a testbed, we optimize the internal surrogate accuracy continues climbing with budget, indicat- GSUAP module using varying subsets of the training data, ing persistent information leakage through perturbed outputs.

ranging from 10 to 500 samples, across channel ratios from *LymphNode* suppresses surrogate accuracy to 13.71–15.28% 20% to 100%. It is crucial to note that the maximum budget across all budgets—near random guessing (10%)—with no of 500 samples constitutes merely **1%** of the original dataset, improvement as budget scales by 50*×*, confirming that feature- simulating an extremely low-resource setup. We evaluate per- space corruption renders collected soft labels uninformative for formance on the test set using the **Accuracy** **Gap** and the distillation.

**Fooling** **Rate**. The Fooling Rate measures the proportion of *Model* *Inversion.* We adopt the GMI framework [16], where a clean images that are misclassified by the target network solely DCGAN trained on public face images is optimized in latent due to the injected perturbation, defined as:

space to reconstruct private training samples of a CelebA Fooling Rate= *N*misclassified with GSUAP classifier. To comply with the oracle-access constraint, we (8) *N*correctly classified clean replace white-box gradients with Natural Evolution Strategies (NES), yielding a fully black-box variant. The attack targets 50 In our visualization, solid and dashed lines distinguish these randomly selected identities. Attack success is measured by an two metrics. We benchmark against the SUAP baseline for independent evaluator: Acc-1/5 (lower is better) indicates how comparison, while Gaussian noise is excluded as prior exper- often reconstructions match the target identity; KNN Distance iments (Sec. IV-A) have already demonstrated its insufficient (higher is better) quantifies feature-space divergence from real neutralizing capability. The results are presented in Figure 5.

samples. Results appear in Table V.

The results in Figure 5 demonstrate the superior data effi- ciency of *LymphNode*. We observe a significant performance TABLE V gap between our method and the SUAP baseline. GSUAP GMI MODEL INVERSION ATTACK RESULTS ONCELEBA (50 IDENTITIES).

exhibits a rapid convergence trajectory, establishing a robust lockout with minimal data. Specifically, at channel ratios of Acc-1*↓* Acc-5*↓* KNN Dist*↑* Defense Latency 40% and above, GSUAP achieves a high Accuracy Gap and None (Clean)

82. 71%
91. 32%
5. 57
0. 082ms

Fooling Rate with as few as 50 to 100 samples. Furthermore, at CIP

44. 30%
48. 23%
6. 45
0. 110ms

higher channel ratios (60%and80%), the protection capability **3.02%** **8.77%** **7.71** PP

0. 308ms

saturates quickly, reaching peak performance with a negligible **0.088ms** Ours

4. 17%
11. 22%
7. 51

data budget (approx. 50 samples). These observations confirm that*LymphNode*entails low setup costs and is highly practical CIP provides limited defense (Acc-1 = 44.30%). PP for real-world deployment.

achieves the strongest inversion protection (Acc-1 = 3.02%) *B.* *Domain-Adaptivity* *Analysis* but performs per-query PGD optimization, increasing latency In practical scenarios, the model provider might need to from 0.082 ms to 0.308 ms—a 275% overhead. CIP incurs a protect a model without accessing its original training set.

moderate 34% increase (0.110 ms) via per-query reliability To evaluate whether *LymphNode* can be effectively deployed scoring. *LymphNode* achieves comparable protection (Acc-1

<!-- Page 10 -->

Channel Ratio: 20% Channel Ratio: 40% Channel Ratio: 60% Channel Ratio: 80% Accuracy Gap (%) Fooling Rate (%) 20 20 20 20 20 20 20 20 0 0 0 0 0 0 0 0 10 20 50 200 500 10 20 50 200 500 10 20 50 200 500 10 20 50 200 500 Calibration Size Calibration Size Calibration Size Calibration Size SUAP Accuracy Gap GSUAP Accuracy Gap SUAP Fooling Rate GSUAP Fooling Rate Fig. 5. effectiveness with different dataset size.

surrogate dataset aligns with the target in basic visual statistics, TABLE VI CROSS-DATASETEFFECTIVENESS: GSUAP GENERATED FROMCIFAR-10 the *LymphNode* framework can establish a robust lockout (SOURCE) APPLIED TO MODELS TRAINED ON DISTINCTTARGET mechanism. This finding is pivotal for practical application, as DOMAINS.

it significantly lowers the barrier for adoption. It validates that model owners can reliably initialize the protection mechanism **CIFAR-10** **(Source)** **CIFAR-100** **(Target)** **STL-10** **(Target)** **Ratio** using publicly accessible surrogate datasets, thereby securing **Gap** **FR** **Gap** **FR** **Gap** **FR** their intellectual property even in scenarios where the original

0. 2
19. 34
22. 22
40. 82
43. 77
14. 70
20. 17

training data is private, lost, or computationally too expensive

0. 4
82. 79
88. 94
69. 98
64. 13
38. 35
42. 58

to process.

0. 6
82. 89
89. 07
70. 66
75. 02
46. 06
50. 20
0. 8
82. 71
88. 68
70. 65
75. 14
51. 31
55. 11

All metrics reported in percentage (%). **Gap**: Accuracy Gap; **FR**: Fooling Rate.

VI. ROBUSTNESSEVALUATION We strictly evaluate the dependability of*LymphNode*against a knowledgeable adversary whose objective is either to forge using accessible proxy datasets, we conducted transferability valid credentials or to physically remove the protection mech- experiments by optimizing GSUAP on the CIFAR-10 dataset anism. In this evaluation, we assume a “deployment-ready” (Source) and deploying the generated perturbation to neutral- threat model where the underlying system integrity is secured, ize ResNet-18 models trained on different target domains:

limiting the adversary to adaptive attacks involving traffic CIFAR-100 and STL-10. We selected these targets to isolate interception, credential fabrication, and model modification.

domain characteristics; CIFAR-100 shares the same resolution To demonstrate the superiority of our plugin-based architecture (32*×*32) as the source but differs in semantic granularity, while over traditional parameter-based protections, we implement STL-10 introduces a larger domain shift with disjoint class two representative baselines: BadNets [50] and Blended At- definitions and higher resolution (96*×*96). For evaluation, we tack [51]. We re-purpose these backdoor attacks as baseline randomly sampled 500 clean images from each target dataset access control mechanisms by training models to yield high- and applied the source-derived perturbation, quantifying effi- fidelity inference only upon detecting specific triggers—a visi- cacy via the Accuracy Gap. The results in Table VI reveal ble3*×*3patch for BadNets and an invisible static noise pattern remarkable adaptability, particularly when the domains share (*α* = 0*.*1) for Blended Attack. This setup allows for a rig- statistical similarities. Specifically, the CIFAR-10 derived per- orous comparative analysis. Our evaluation comprehensively turbation achieves a control efficacy on the CIFAR-100 target covers four critical dimensions: watermark imperceptibility, that closely trails the oracle setting at higher channel ratios.

robustness against credential forgery, resilience to fine-tuning Notably, at the lowest sparsity (0.2), the cross-domain attack attacks, and stability under lossy compression and pruning.

on CIFAR-100 (40*.*82% Gap) significantly outperforms the source attack on CIFAR-10 (19*.*34% Gap). We attribute this *A.* *Watermark* *Imperceptibility* *Analysis* to the inherent fragility of the fine-grained CIFAR-100 model, where the coarse-grained adversarial patterns learned from To ensure authorization credentials remain undetectable to CIFAR-10 prove disruptively potent against the more complex human or automated inspection, we evaluate visual stealth- decision boundaries. While efficacy naturally degrades on the iness on 1,000 randomly sampled CIFAR-10 image pairs more distant STL-10 domain due to resolution mismatches, using PSNR, SSIM, and Learned Perceptual Image Patch functional control remains intact.

Similarity (LPIPS). While standard metrics focus on pixel- Synthesizing these observations, we conclude that access to level differences, LPIPS measures perceptual distance in deep the original training dataset is not a prerequisite for *LymphN-* feature space, serving as a critical indicator for high-frequency *ode*. The strong transferability demonstrates that as long as the artifacts that might escape traditional metrics.

<!-- Page 11 -->

Clean Accuracy Recovery During Fine-tuning Attack TABLE VII QUANTITATIVE COMPARISON OF WATERMARK IMPERCEPTIBILITY ON

|Col1|Col2|Col3|Col4|56|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||52<br>53<br>54<br>55<br><br>Acc (%)||||||
|||||52<br>53<br>54<br>55<br><br>Acc (%)||||||
|||||52<br>53<br>54<br>55<br><br>Acc (%)||||||
|||||52<br>53<br>54<br>55<br><br>Acc (%)||||||
|||||38<br>40<br><br>50<br>51||||||
|||||38<br>40<br><br>50<br>51||||||
|||||38<br>40<br><br>50<br>51|||42<br>44<br>46<br>48<br>Epoch|42<br>44<br>46<br>48<br>Epoch|50|
|||||||||||
|||||||||||
|||||||||||
|||||||||LymphN<br>BadNets<br>Blended<br>Passpor|ode<br><br><br>t|

CIFAR-10.*↑*INDICATES HIGHER IS BETTER;*↓*INDICATES LOWER IS BETTER.

**PSNR** **(dB)** *↑* **SSIM** *↑* **LPIPS** *↓* **Method** Test Accuracy (%) 28*.*3502 0*.*9370 0*.*0472 BadNets 33*.*9331 0*.*8245 0*.*0218 Blended **56***.***6411** **0***.***9990** **0***.***0011** **LymphNode** 20 The quantitative results in Table VII highlight the superior fidelity of our approach. The baselines exhibit clear limita- 0 tions: BadNets suffers from low PSNR (28.35 dB) due to 0 10 20 30 50 Fine-tuning Epoch concentrated corruption, while the Blended Attack degrades structural quality (SSIM 0.8245) through global noise in- Fig. 6. Robustness against fine-tuning attacks. The trajectories illustrate the jection. In stark contrast, LymphNode achieves near-perfect Clean Accuracy recovery over 50 epochs.

imperceptibility across all metrics, attaining a PSNR of 56.64 dB and an LPIPS score of 0.0011—an order of magnitude approximation, our discrete feature-bit alignments will resist superior to the baselines. This stealthiness is fundamentally such replication.

attributed to our precision-targeted embedding strategy, which The quantitative results in Table VIII validate this hy- targets deep bit-depths (*s**≥*5) to restrict feature modifications pothesis, revealing distinct vulnerability patterns. The Bad- to the level of numerical quantization noise, thereby preserving Nets baseline yields a 100% forgery success rate due to its both pixel statistics and deep perceptual integrity.

simple trigger, while the Blended Attack remains vulnerable *B.* *Robustness* *against* *Credential* *Forgery* to U-Net (23.8% success rate), indicating that its reliance on continuous feature activations allows for approximation by generative models given sufficient data. In stark contrast, TABLE VIII *LymphNode* achieves absolute robustness with a 0.0% success ROBUSTNESS AGAINSTCREDENTIALFORGERY ATTACKS. WE REPORT THE*Forgery* *Success* *Rate*(SUCCESS RATE OF FORGED CREDENTIALS) ON rate against all attempts. This superior security is attributed THE PROTECTED MODELS. A LOWER ACCURACY INDICATES HIGHER to the fundamental verification discontinuity inherent in our ROBUSTNESS AGAINST FORGERY.

design; while generative models like U-Net inherently pro- duce smooth outputs with microscopic floating-point residuals, **Target** **Model** **Attack** **Method** **Forgery** **Success** **Rate.** **(%)** these deviations disrupt the precise quantization logic required Linear Est.

100. 0

for our LSB-based verification. Consequently, our discrete BadNets (Baseline) U-Net

100. 0

feature-domain mechanism establishes a mathematical barrier Linear Est.

0. 0

that renders forged credentials invalid, proving intractable for Blended (Baseline) U-Net

23. 8

standard continuous approximation algorithms.

**0.0** Linear Est.

**Ours** **(LymphNode)** *C.* *Robustness* *against* *Fine-tuning* *Attacks* **0.0** U-Net Beyond credential forgery, an adversary possessing the model artifacts might attempt to attack the protection mech- To evaluate robustness against credential forgery, we simu- anism through fine-tuning. We simulate a gray-box scenario late an adversary who attempts to reverse-engineer the autho- where the adversary has access to the model parameters and rization watermark using 500 intercepted pairs of unauthorized clean training data (10%), but lacks knowledge of the specific and authorized images. We benchmark our method against plugin logic. To evaluate resilience, we benchmark against the Blended-based model as a primary competitor due to its BadNets, Blended Attack, and Passport [52] under identical use of invisible static triggers. The evaluation employs two conditions. The adversary employs standard Stochastic Gradi- approximation attacks targeting the mathematical properties ent Descent (SGD) with a learning rate of 0*.*01 to fine-tune of the watermark: Linear Residual Estimation, which assumes the protected models. The fine-tuning process is conducted for a static additive signal derived by averaging residuals to sup- 50 epochs using the clean CIFAR-10 dataset.

press image-specific variations, and a Deep Mapping Attack Figure 6 illustrates the accuracy recovery trajectories. We via U-Net, which treats forgery as a supervised image-to- observe that after 50 epochs, the clean accuracy of the *Lym-* image translation task minimizing pixel-wise reconstruction *phNode*-protected model recovers to 52*.*46%, which remains loss. We quantify robustness using the Forgery Success Rate significantly below the original fidelity (*≈*95%). This final on forged images, hypothesizing that while baselines relying accuracy is comparable to that of the structural baseline on continuous pixel-domain modifications are susceptible to

<!-- Page 12 -->

(Passport,54*.*89%) and the data-poisoning baselines (*≈*53%).

VII. DISCUSSION ANDFUTUREWORK These results indicate that *LymphNode*, despite being a post- While *LymphNode* demonstrates efficient and robust per- hoc plugin, achieves a level of resistance against fine-tuning formance, we acknowledge two specific limitations in the that is on par with heavyweight structural defenses like Pass- current implementation. First, as a software-level plugin, the port, effectively preventing the restoration of model utility defense relies on the integrity of the hosting runtime envi- under standard fine-tuning protocols.

ronment; an adversary with unrestricted write access to the *D.* *Resilience* *to* *Lossy* *Compression* model parameters could theoretically identify and bypass the protection logic. Future work can address this by coupling In practical deployment scenarios, input images frequently the plugin with lightweight code obfuscation or hardware- undergo lossy compression during transmission, such as JPEG backed integrity checks to withstand physical tampering. Sec- encoding, which quantizes high-frequency DCT coefficients ond, the current verification mechanism relies on fine-grained and inherently challenges the precision-based LSB verification feature perturbations optimized for floating-point models. In mechanism. To evaluate the robustness of our system against scenarios employing aggressive quantization (e.g., INT8) for such channel distortions, we measured the Authorization Suc- extreme compression, these subtle credentials may be distorted cess Rate (ASR) of credentials subjected to JPEG compression or lost. Future works can extend the framework to support across varying quality factors. Recognizing the sensitivity of quantization-aware embedding strategies compatible with low- standard single-step embedding to aggressive quantization, we precision arithmetic.

adopted an iterative embedding strategy that progressively refines pixel perturbations to survive the specific quantization VIII. CONCLUSION matrix of the target quality level.

In this paper, we introduced *LymphNode*, a novel post- hoc plugin framework designed to secure Deep Neural Net- TABLE IX AUTHORIZATIONSUCCESSRATE(ASR) EVOLUTION USING THE works against model extraction and inversion attacks. Ad- ITERATIVEEMBEDDING STRATEGY [37].

dressing the critical vulnerability of unrestricted oracle access, the framework establishes an immunological “default-deny”

|Quality (Q)|Iterations (T)|
|---|---|
|**Quality (**_Q_**)**|10<br>20<br>30<br>40<br>50|

checkpoint. It actively neutralizes model utility via General- ized Sparse Universal Adversarial Perturbations (GSUAP), ef-

|80<br>70<br>60|15.6% 82.4% 96.5% 99.1% 99.8%<br>2.3% 45.8% 81.2% 93.7% 98.4%<br>0.0% 8.5% 35.2% 62.8% 85.6%|
|---|---|

fectively blocking the gradient estimation required for extrac- tion, while transparently restoring fidelity for authorized users through a discrete feature-domain verification mechanism.

Specifically, its ability to initialize robust protection using only The performance evolution, detailed in Table IX, demon- public surrogate datasets—combined with a strictly constant strates the efficacy of this adaptive approach. While the *O*(1) inference overhead—ensures that the framework can survival rate is initially negligible under severe compression be seamlessly scaled and instantly deployed across hetero- (e.g., quality factor 60), it improves dramatically to over 85% geneous, resource-constrained edge nodes without hardware- after 50 iterations. These results confirm that *LymphNode* specific tuning.

can be effectively hardened to guarantee reliable access for Our comprehensive evaluation confirms that *LymphNode* authorized users even in bandwidth-constrained environments, successfully reconciles rigorous security with operational prac- successfully balancing strict security requirements with prac- ticality. We demonstrated that the protection is exceptionally tical usability.

data-efficient, establishing robust neutralization with as few as 50–100 calibration samples (*<* 1% of training data) that *E.* *Resistance* *to* *Model* *Pruning.* need not originate from the original dataset, thus eliminating While we focus on fine-tuning, model pruning represents the dependency on sensitive private data required by prior art.

another potential removal vector. However, our framework in- From a system perspective, the plugin introduces negligible herently mitigates this threat through the design of the**Weight-** overhead (*≈*1 ms latency), ensuring viability for real-time **Gradient** selector (evaluated in Sec. IV-D). As demonstrated applications. Furthermore, robustness analysis verifies that our in Fig. 3, the protection mechanism is anchored to the channels credentials achieve near-perfect imperceptibility and exhibit with the highest gradient sensitivity—i.e., the most “decision- strong resilience against adaptive threats, including generative critical” features of the network. Standard pruning algorithms, forgery and fine-tuning. By decoupling IP protection from the which eliminate redundant neurons [41], would naturally constraints of model retraining, *LymphNode* offers a scalable, bypass our plugin-associated channels. Conversely, an ag- deployment-ready solution for safeguarding high-value AI gressive adversary attempting to prune these specific high- assets.

saliency channels would inadvertently destroy the model’s primary classification capability before successfully removing REFERENCES the protection. This establishes a **structural** **coupling** where [1] OpenAI, “Gpt-4 technical report,” OpenAI, Tech.

Rep., 2024, the survival of the security mechanism is tied to the utility of arXiv:2303.08774v6.

[Online].

Available:

https://arxiv.org/abs/2303.

the model itself, rendering pruning attacks ineffective.

08774

<!-- Page 13 -->

[21] E. Le Merrer, P. P´erez, and G. Tr´edan, “Adversarial frontier stitching [2] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, for remote neural network watermarking,” *Neural* *Computing* *and* *Ap-* T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, *plications*, vol. 32, no. 13, pp. 9233–9244, 2020.

A. Joulin, E. Grave, and G. Lample, “Llama: Open and efficient foundation language models,” *arXiv* *preprint* *arXiv:2302.13971*, 2023.

[22] A. Chakraborty *et* *al.*, “DynaMarks: Defending against deep learn- [Online]. Available: https://arxiv.org/abs/2302.13971 ing model extraction using dynamic watermarking,” *arXiv* *preprint* [3] F. Tram`er, F. Zhang, A. Juels, M. K. Reiter, and T. Ristenpart, “Stealing *arXiv:2207.13321*, 2022.

machine learning models via prediction APIs,” in*25th USENIX Security* [23] N. Lukas, E. Jiang, X. Li, and F. Kerschbaum, “Sok: How robust is *Symposium* *(USENIX* *Security* *16)*.

USENIX Association, 2016, pp.

image classification deep neural network watermarking?” in *2022* *IEEE* 601–618.

*Symposium* *on* *Security* *and* *Privacy* *(SP)*.

IEEE, 2022, pp. 787–804.

[4] M. Jagielski, N. Carlini, D. Berthelot, A. Kurakin, and N. Papernot, [24] X. Liu, Y. Zhong, Y. Zhang, L. Qin, and W. Deng, “Enhancing “High accuracy and high fidelity extraction of neural networks,” in generalization of universal adversarial perturbation through gradient *29th* *USENIX* *Security* *Symposium* *(USENIX* *Security* *20)*.

USENIX aggregation,” in*Proceedings of the IEEE/CVF International Conference* Association, 2020, pp. 1345–1362.

*on* *Computer* *Vision* *(ICCV)*, 2023, pp. 4428–4437.

[5] M. Fredrikson, S. Jha, and T. Ristenpart, “Model inversion attacks [25] Y. Zhang, Y. Xu, J. Shi, L. Y. Zhang, S. Hu, M. Li, and Y. Zhang, that exploit confidence information and basic countermeasures,” in “Improving generalization of universal adversarial perturbation via dy- *Proceedings* *of* *the* *22nd* *ACM* *SIGSAC* *Conference* *on* *Computer* *and* namic maximin optimization,” in *Proceedings* *of* *the* *AAAI* *Conference* *Communications* *Security* *(CCS)*.

ACM, 2015, pp. 1322–1333.

*on* *Artificial* *Intelligence*, vol. 39, 2025, pp. –.

[6] Y. Adi, C. Baum, M. Cisse, B. Pinkas, and J. Keshet, “Turning [26] C. Xu and G. Singh, “Robust universal adversarial perturbations,”*arXiv* your weakness into a strength: Watermarking deep neural networks by *preprint* *arXiv:2206.10858*, 2022.

backdooring,” in *27th* *USENIX* *Security* *Symposium* *(USENIX* *Security* [27] X. Zhong and C. Liu, “Sparse-PGD: A unified framework for sparse *18)*, 2018, pp. 1615–1631.

adversarial perturbations generation,” *arXiv* *preprint* *arXiv:2405.05075*, [7] Y. Uchida, Y. Nagai, S. Sakazawa, and S. Satoh, “Embedding water-

2024. 

marks into deep neural networks,” in *Proceedings* *of* *the* *2017* *ACM* *on* [28] K. R. Mopuri, A. Ganeshan, and R. V. Babu, “Generalizable data- *International* *Conference* *on* *Multimedia* *Retrieval*, 2017, pp. 269–277.

free objective for crafting universal adversarial perturbations,” in *IEEE* [8] J. Zhang, Z. Gu, J. Jang, H. Wu, M. P. Stoecklin, H. Huang, and *Transactions* *on* *Pattern* *Analysis* *and* *Machine* *Intelligence* *(TPAMI)*, I. Molloy, “Protecting intellectual property of deep neural networks vol. 41, no. 10, 2019, pp. 2452–2465.

with watermarking,” in *Proceedings* *of* *the* *2018* *on* *Asia* *Conference* [29] A. Chakraborty, A. Mondal, and A. Srivastava, “Hardware-assisted *on* *Computer* *and* *Communications* *Security*, 2018, pp. 159–172.

intellectual property protection of deep learning models,” in *2020* *57th* [9] M. Alam, S. Saha, D. Mukhopadhyay, and S. Kundu, “Deep-lock: Secure *ACM/IEEE* *Design* *Automation* *Conference* *(DAC)*.

IEEE, 2020, pp.

authorization for deep neural networks,” in *2020* *IEEE* *38th* *VLSI* *Test* 1–6.

*Symposium* *(VTS)*.

IEEE, 2020, pp. 1–6.

[30] Y. Gong, D. Chen, W. Niu, S. Cheng, X. Pan, Q. Nie, Y. Xiao, L. Zhang, [10] H. Chen, C. Fu, J. Zhao, and F. Koushanfar, “Model assertion: A defense and H. Zheng, “Modellock: Locking your model with a spell,” in against model theft via authorized model encryption,” in *Proceedings of* *Proceedings* *of* *the* *32nd* *ACM* *International* *Conference* *on* *Multimedia*, *the* *IEEE/CVF* *International* *Conference* *on* *Computer* *Vision* *(ICCV)*, 2024, pp. 6595–6604.

2021, pp. 15 380–15 389.

[31] F. Tramer and D. Boneh, “Slalom: Fast, verifiable and private execution [11] M. Xue, Y. Wu, L. Y. Zhang, D. Gu, Y. Zhang, and W. Liu, “Ssat:

of neural networks in trusted hardware,” in *International* *Conference* *on* Active authorization control and user’s fingerprint tracking framework *Learning* *Representations* *(ICLR)*, 2019.

for dnn ip protection,” *ACM* *Transactions* *on* *Multimedia* *Computing,* [32] D. Natarajan, A. Loveless, W. Dai, and R. Dreslinski, “Chex-mix: Com- *Communications* *and* *Applications*, vol. 20, no. 10, 2024.

bining homomorphic encryption with trusted execution environments [12] M. Xue, Z. Wu, J. Wang, Y. Zhang, and W. Liu, “Advparams: An active for two-party oblivious inference in the cloud,” in *8th* *IEEE* *European* dnn intellectual property protection technique via adversarial pertur- *Symposium* *on* *Security* *and* *Privacy* *(EuroS&P)*, 2023, pp. 457–477.

bation based parameter encryption,” *arXiv* *preprint* *arXiv:2105.13697*, [33] M. Alam, S. Saha, D. Mukhopadhyay, and S. Kundu, “Nn-lock: A

2021. 

lightweight authorization to prevent ip threats of deep learning models,” [13] T. Orekondy, B. Schiele, and M. Fritz, “Prediction poisoning: Towards *ACM Journal on Emerging Technologies in Computing Systems*, vol. 18, defenses against DNN model stealing attacks,” in *International* *Confer-* no. 2, pp. 1–27, 2022.

*ence* *on* *Learning* *Representations* *(ICLR)*, 2020.

[34] W. Feng *et* *al.*, “Survey of research on confidential computing,” *IET* [14] H. Zhang, G. Hua, X. Wang, H. Jiang, and W. Yang, “Categorical *Communications*, vol. 18, no. 8, pp. 465–486, 2024.

inference poisoning: Verifiable defense against black-box DNN model [35] M. Xue, S. Sun, C. He, D. Gu, Y. Zhang, J. Wang, and W. Liu, stealing without constraining surrogate data and query times,” *IEEE* “Activeguard: An active intellectual property protection technique for *Transactions* *on* *Information* *Forensics* *and* *Security*, vol. 18, pp. 1473– deep neural networks by leveraging adversarial examples as users’ 1486, 2023.

fingerprints,” *IET* *Computers* *&* *Digital* *Techniques*, vol. 17, no. 3-4, [15] T. Orekondy, B. Schiele, and M. Fritz, “Knockoff nets: Stealing func- pp. 111–126, 2023.

tionality of black-box models,” in *CVPR*, 2019.

[36] M. Jiang *et* *al.*, “AMAO: A comprehensive defense framework against [16] Y. Zhang, R. Jia, H. Pei, W. Wang, B. Li, and D. Song, “The secret model extraction attacks,”*IEEE Transactions on Dependable and Secure* revealer: Generative model-inversion attacks against machine learning *Computing*, vol. 21, no. 2, 2024.

models,” in *Proceedings* *of* *the* *IEEE/CVF* *Conference* *on* *Computer* [37] Z.

Liu, F.

Li, Z.

Li, and B.

Luo, “Loneneuron:

A highly- *Vision* *and* *Pattern* *Recognition* *(CVPR)*, 2020, pp. 253–261.

effective feature-domain neural trojan using invisible and polymorphic [17] B. D. Rouhani, H. Chen, and F. Koushanfar, “Deepsigns: An end-to-end watermarks,” in *Proceedings* *of* *the* *2022* *ACM* *SIGSAC* *Conference* watermarking framework for ownership protection of deep neural net- *on* *Computer* *and* *Communications* *Security*, ser. CCS ’22.

New works,” in *Proceedings* *of* *the* *Twenty-Fourth* *International* *Conference* York, NY, USA: ACM, 2022, pp. 2129–2143. [Online]. Available:

*on* *Architectural* *Support* *for* *Programming* *Languages* *and* *Operating* https://dl.acm.org/doi/10.1145/3548606.3560678 *Systems* *(ASPLOS)*.

ACM, 2019, pp. 485–497.

[38] S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, “Univer- [18] Y. Li, Y. Bai, Y. Jiang, Y. Yang, S.-T. Xia, and B. Li, “Untargeted sal adversarial perturbations,” in*Proceedings of the IEEE conference on* backdoor watermark: Towards harmless and stealthy dataset copyright *computer* *vision* *and* *pattern* *recognition*, 2017, pp. 1765–1773.

protection,” in *Advances* *in* *Neural* *Information* *Processing* *Systems* [39] P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz, “Pruning *(NeurIPS)*, vol. 35, 2022, pp. 13 862–13 875, oral, Top 2%.

convolutional neural networks for resource efficient inference,” in *In-* [19] J. Guo, Y. Li, L. Wang, S.-T. Xia, H. Huang, C. Liu, and B. Li, *ternational* *Conference* *on* *Learning* *Representations*, 2017.

“Domain watermark: Effective and harmless dataset copyright protection [40] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, “Pruning is closed at hand,” in *Advances* *in* *Neural* *Information* *Processing* filters for efficient convnets,” in *International* *Conference* *on* *Learning* *Systems* *(NeurIPS)*, 2023.

*Representations* *(ICLR)*, 2017.

[20] X. Cao, J. Jia, and N. Z. Gong, “IPGuard: Protecting intellectual property [41] S. Han, J. Pool, J. Tran, and W. J. Dally, “Learning both weights of deep neural networks via fingerprinting the classification boundary,” in *Proceedings* *of* *the* *2021* *ACM* *Asia* *Conference* *on* *Computer* *and* and connections for efficient neural networks,” in *Advances* *in* *Neural* *Communications* *Security* *(AsiaCCS)*, 2021, pp. 14–25.

*Information* *Processing* *Systems* *(NeurIPS)*, 2015, pp. 1135–1143.

<!-- Page 14 -->

[42] A. Krizhevsky, G. Hinton *et* *al.*, “Learning multiple layers of features from tiny images,” University of Toronto, Tech. Rep., 2009.

[43] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” *Proceedings* *of* *the* *IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.

[44] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading digits in natural images with unsupervised feature learning,” in *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*,

2011. 

[45] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in*Proceedings of the IEEE conference on computer vision* *and* *pattern* *recognition*, 2016, pp. 770–778.

[46] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly *et* *al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *International* *Conference* *on* *Learning* *Representations*, 2021.

[47] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in *Advances* *in* *Neural* *Infor-* *mation* *Processing* *Systems*, vol. 25, 2012.

[48] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in *Proceedings* *of* *the* *IEEE* *Confer-* *ence* *on* *Computer* *Vision* *and* *Pattern* *Recognition* *(CVPR)*, 2017, pp.

4700–4708.

[49] P. Molchanov, A. Mallya, S. Tyree, I. Frosio, and J. Kautz, “Importance estimation for neural network pruning,” in*Proceedings of the IEEE/CVF* *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 11 264–11 272.

[50] T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnera- bilities in the machine learning model supply chain,” *arXiv* *preprint* *arXiv:1708.06733*, 2017.

[51] X. Chen, C. Liu, B. Li, K. Lu, and D. Song, “Targeted backdoor attacks on deep learning systems using data poisoning,” *arXiv* *preprint* *arXiv:1712.05526*, 2017.

[52] L. Fan, K. W. Ng, and C. S. Chan, “Deepipr: Deep neural network ownership verification with passports,” *IEEE* *Transactions* *on* *Pattern* *Analysis and Machine Intelligence*, vol. 44, no. 10, pp. 6122–6139, 2022.
