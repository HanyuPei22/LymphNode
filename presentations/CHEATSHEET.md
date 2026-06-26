# LymphNode — 17-min Talk Cheatsheet

Total ≈ 17 min. ~21 content slides + 4 backup. Numbers to never get wrong are in **bold**.
Arc: *Why (motivation) → who's tried (related work) → our idea & how (design) → does it work (results) → honest scope.*

---

## 0. Title + Outline  — 0:30
- "LymphNode: a plug-and-play **access-control** plugin for deployed DNNs."
- One breath on outline; don't read it.

---

## MOTIVATION  (~2:00)

### 1. Security Concerns for DNN Deployment  — 1:00
- **Hook:** DNNs are expensive IP, but we ship them to the edge/on-prem → attacker gets **unlimited, zero-latency, un-rate-limited oracle access**.
- FDA chart = AI models are being deployed for real, fast (healthcare/finance can't use cloud APIs).
- Three attack types: **extraction** (clone via queries), **inversion** (rebuild private training data from outputs), **adaptive probing**.

### 2. Attack Surface and Defense Surface  — 1:00
- Left = where attackers get in (query behavior, confidence outputs, the shipped file).
- Right = the 4 places defenders can intervene → maps 1:1 to the families on the next slide.
- **Punchline:** "existing defenses differ mainly in *where* they intervene" → sets up the gap table.
- Note: this is the *general* landscape; our actual target is narrowed on the Threat Model slide. Don't over-claim here.

---

## RELATED WORK  (~1:15)

### 3. Existing Defenses at a Glance — and the Gap  — 1:15
- Walk the rows, one line each:
  - **Watermark/fingerprint** → passive, only proves ownership *after* theft (✗ prevent use).
  - **Output poisoning (PP/CIP)** → needs a **server** to filter each query → gone at the edge (✗ edge).
  - **Crypto/HW lock** → special hardware or per-query crypto → **high cost** (✗ edge).
  - **Train-time lock** → needs **original data and/or retraining** (✗ no-data, ✗ no-retrain).
- **Punchline:** "LymphNode is the only row that's all-✓." That's the whole positioning.

---

## LYMPHNODE DESIGN  (~6:00)

### 4. LymphNode in One Idea  — 0:45
- **The 30-sec version:** model is **default-deny** — every input hits a sparse feature-space perturbation that kills utility; only an input carrying a hidden **feature-domain credential** triggers an antidote that restores full fidelity.
- "Immune system inside the model." Read the 5 tags and move on.

### 5. Threat Model  — 0:45
- 3 honest parties (owner / edge operator / authorized user).
- **Adversary = gray-box, oracle-only:** knows architecture + mechanism; **cannot** see key, intermediate features, or gradients; only weapon = unlimited queries.
- Punchline: "no server to filter queries → protection must live *inside* the forward pass."

### 6. System Overview  — 1:00
- Walk the framework figure left→right: input → early conv features → **checkpoint reads 32 bits** → branch.
- No key → GSUAP stays on (deny). Valid key → antidote restores.
- Emphasize: **frozen backbone, no retraining, element-wise ops only.**

### 7. Component 1 — Feature-Domain Credential (figure first)  — 0:50
- Lead with the figure; walk ①→⑥: input → conv-1 features → read bit at **s=6** → compare to **32-bit** key → flip only that deep bit (red) → invert to pixels (tiny w).
- Punchline: change lives at quantization-noise level → **invisible (PSNR 56.6 dB)** but an exact discrete match on verification.

### 8. Component 1 — Generation & Collision  — 0:55
- **Key:** 32 bits over **v=4** kernels, bit b=⌊|y|·2^s⌋ mod 2, s=6.
- **Generation:** find tiny pixel w by random search so features match k; O(h·2^v) → **1000 creds in <2 s**.
- **Verification:** re-extract 32 bits, compare — cheap discrete check, no optimization.
- **Collision analysis:** benign input matches all 32 bits only by chance, **P_c = 2⁻³² ≈ 2.3×10⁻¹⁰** → default-deny is *sound*, no accidental authorization. (Active forgery resistance → Result 6.)

### 9. Component 2 — GSUAP Neutralization  — 1:15  ⭐ core method
- Two phases: (1) **rank channels** by gradient sensitivity, keep top-k "decision-critical" (mask M); (2) optimize one universal Δ by **projected gradient ASCENT** to maximize CE loss, ℓ∞≤ε.
- **vs UAP/SUAP (the key contrast, expect a question):** plain UAP spreads its budget over *all* channels; SUAP just masks a generic UAP afterward (throws most of it away). GSUAP optimizes Δ *inside* the selected channels → concentrates the full budget on the model's command points → that's why the gap is huge on robust ResNets/ViTs.
- (If asked about GD-UAP: we borrow its *data-light/generalizable* spirit; our loss is CE-max on a tiny calibration set, see code `uap_trainer.py`.)

### 10. Inference Logic (tikz)  — 0:30
- Same frozen backbone, two paths; only the feature treatment changes after verification. Quick.

---

## EXPERIMENTAL RESULTS  (~6:00)

### 11. Experimental Setup  — 0:45
- Datasets (CIFAR-10/MNIST/SVHN + transfer + CelebA), 6 architectures, baselines, metrics.
- Frame it as the **5 questions**: lock out? at what cost? real attacks? how little data? how robust? — these are Results 1–6.

### 12. Result 1 — Lockout  — 0:50
- ResNet-18/CIFAR-10 @60%: **Gauss 85.4 / SUAP 72.0 / GSUAP 13.6**, VIP 94.5.
- Story: GSUAP → near random, monotone in ratio, VIP untouched. Gaussian fails on robust nets. Full 6×3 sweep shown.

### 13. Result 2 — Efficiency  — 0:40
- E = gap / ρ. GSUAP **≈2.5 @20%** on ResNet-18 vs Gaussian <0.8; gap widens on robust archs. (gradient-guided beats stochastic.)

### 14. Result 3 — Overhead  — 0:45
- Params/FLOPs **unchanged**; latency **+~1 ms constant** (1.70→2.74 / 1.54→2.60); throughput −14.8% / −6.7%.
- Punchline: no per-query optimization → constant cost; fits 30 FPS (33 ms) easily.

### 15. Result 4 — Real Attacks  — 1:00
- **Extraction (KnockoffNets):** no-def **85.2%** @50K → PP/CIP 45.7/48.3 → **LymphNode 15.3%**, and *flat* across 1K–50K (soft labels carry no signal).
- **Inversion (GMI):** clean Acc-1 82.7 → LymphNode **4.2%**; matches PP's protection but **0.088 ms vs PP 0.308 ms** (PP pays per-query PGD).

### 16. Result 5 — Data Efficiency (figure)  — 0:40
- Robust lockout from **50–100 samples (<1%)**; ≥40% ratio saturates fast, beats SUAP.
- Define if asked: **FR = fraction of correctly-classified clean images the perturbation flips.**

### 17. Result 5 — Data-Free Transfer  — 0:45
- GSUAP made on CIFAR-10, applied to other domains: CIFAR-100 gap **~70%** (near the in-domain oracle 82.9%); STL-10 degrades gracefully under resolution shift.
- Punchline: **no original data needed — a public surrogate works.**

### 18. Result 6 — Stealth & Forgery  — 0:40
- Imperceptibility **PSNR 56.6 / SSIM 0.999 / LPIPS 0.001** (≫ BadNets/Blended).
- Forgery **0.0%** for both linear & U-Net (vs Blended 23.8% U-Net) — discrete LSB barrier.

### 19. Result 6 — Resistance to Removal  — 0:45
- Fine-tune 50 ep recovers only **52.5%** (vs orig ~95%), on par with heavyweight Passport (54.9%) — but we're post-hoc.
- JPEG-Q60 → **85.6%** auth via iterative embedding (full sweep in backup). Pruning kills utility first (structural coupling).

---

## SCOPE & TAKEAWAY  (~1:15)

### 20. Limitations and Honest Scope  — 0:35
- Honest limits: needs an early **conv layer**; needs **image inputs**.
- Future: extend default-deny to **LLMs / other modalities**; HW-backed integrity + obfuscation; quantization-aware embedding.

### 21. Takeaway  — 0:40
- **One line:** "LymphNode shifts DNN IP protection from proof-*after*-theft to prevention-*before*-use."
- Recap: feature-space default-deny; no retrain, <1% data; ~1 ms O(1); beats PP/CIP, resists forgery/fine-tune/compression. → "Thank you / questions."

---

## BACKUP (Q&A only — know which slide answers which question)
- **"Can you tune severity?"** → Backup: Noise Scale λ (VIP flat ~99.8%, unauthorized declines smoothly).
- **"Why weight-gradient selection?"** → Backup: Channel Selector Ablation (beats Taylor/WeightNorm/Random; 40–60% ≈ full).
- **"Show the fine-tuning trajectory."** → Backup: Fine-Tuning Robustness curve.
- **"JPEG full numbers?"** → Backup: JPEG table (Q80/70/60 × T=10–50).

## Anticipated hard questions
- **GSUAP vs GD-UAP loss?** → ours = CE-max on tiny calibration set (code `uap_trainer.py`); GD-UAP = activation-saturation, data-free; we borrow its data-light spirit, not its loss.
- **Why not just label-only output?** → inversion (GMI) needs scores; our threat model grants oracle probability access, and we still block it.
- **Isn't this just a train-time lock?** → same *goal* (active access control) but intervention moved **post-hoc into feature space** → that's what removes the retrain/original-data requirement.
- **What if attacker has the weight file?** → covered by robustness tier (Result 6): fine-tune/prune/forge all fail; full white-box parameter tampering is acknowledged future work.
