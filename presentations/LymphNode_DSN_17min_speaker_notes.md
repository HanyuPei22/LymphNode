# LymphNode 17-minute speaker notes

Target pacing: 23 main slides plus one backup slide. Average pace is about 40-45 seconds per main slide, with extra time on framework and main result slides.

## 1. LymphNode (0:00-0:40)
Open with the concrete problem: valuable DNNs must often be deployed outside a controlled cloud.
LymphNode is an access-control layer inside the model, not a watermark used after a theft has
already happened.

## 2. Talk Structure (0:40-1:10)
Set expectations. About one third of the talk is motivation and related work, one third is design,
and one third is results.

## 3. Why This Problem Matters (1:10-1:55)
Frame the conflict: deployment moves models closer to users and attackers. That is good for latency
and privacy, but it weakens cloud-style access control.

## 4. Threat: Oracle Access Enables Model Theft (1:55-2:45)
Use this slide to explain why the attacker does not need weights. The attacker only needs repeated
queries and output probabilities or labels.

## 5. Related Work: Three Defense Families (2:45-3:35)
This is the related-work taxonomy. The important transition is that LymphNode borrows the access-
control objective from active defenses, but tries to remove their setup and deployment friction.

## 6. Where Existing Active Defenses Struggle (3:35-4:20)
Do not over-claim that prior work is useless. The claim is narrower: none of them simultaneously
gives post-hoc deployment, low setup cost, low overhead, and edge compatibility.

## 7. Contribution in One Sentence (4:20-5:00)
This is the anchor sentence. Repeat the two flows: no credential means neutralized output; valid
credential means antidote and clean utility.

## 8. Threat Model and Deployment Roles (5:00-5:45)
Clarify the assumption: LymphNode protects against oracle-access abuse under runtime integrity. It
does not claim to solve arbitrary write access to model files.

## 9. System Overview (5:45-6:40)
Walk left to right. Clean image plus credential becomes authorized input. The checkpoint verifies
feature bits. If matched, inverse noise cancels GSUAP. Otherwise, the target DNN sees corrupted
features.

## 10. Feature-Domain Credential (6:40-7:30)
Emphasize why this is not a BadNets-style visible trigger. The credential lives in a feature
representation and uses fine-grained quantized bits.

## 11. GSUAP: Neutralize by Touching Critical Channels (7:30-8:30)
The method has two phases: decide where to intervene, then learn one universal perturbation. During
inference, the perturbation is just a tensor addition.

## 12. Inference Logic (8:30-9:10)
This slide is useful if the audience missed the figure. Make the decision logic explicit: same
backbone, different feature-space treatment based on credential verification.

## 13. Artifact and Code Mapping (9:10-9:45)
For an artifact-aware DSN audience, this slide helps show that the method is implemented as a clear
pipeline, not just a conceptual figure.

## 14. Experimental Setup (9:45-10:25)
State the evaluation contract before showing results. Lower unauthorized accuracy is better; high
VIP accuracy means normal service for authorized users.

## 15. Main Result: Lock Out Unauthorized Queries (10:25-11:25)
This is the most important quantitative slide. The contrast is simple: naive noise barely locks the
model, SUAP helps, GSUAP makes the model almost useless to unauthorized users while keeping VIP
accuracy at 94.5%.

## 16. Efficiency Frontier (11:25-12:05)
Explain the efficiency metric as protection benefit per channel touched. The point is not only that
GSUAP works, but that it works efficiently.

## 17. Runtime Cost is Constant and Small (12:05-12:45)
This is the practical deployment argument. LymphNode does not run per-query optimization; it adds a
fixed sparse tensor operation.

## 18. Protection Against Real Attack Pipelines (12:45-13:50)
This connects the method back to the opening threat model: if labels and probabilities are corrupted
in feature space, the data collected by extraction and inversion attacks becomes uninformative.

## 19. Low Data Requirement (13:50-14:30)
This is a key differentiator. Prior active defenses often need original data or retraining.
LymphNode can initialize with a very small calibration set.

## 20. Cross-Dataset Adaptivity (14:30-15:05)
If the original data is unavailable, a public surrogate can still give useful protection when visual
statistics are related. Acknowledge STL-10 is harder due to domain and resolution shift.

## 21. Robustness and Stealth (15:05-15:55)
Summarize robustness without drowning in details. The credential is hard to see, hard to forge, and
the lock is not easily removed by normal fine-tuning.

## 22. Limitations and Honest Scope (15:55-16:30)
This slide makes the defense credible. It is an access-control layer under a clear deployment model,
not a universal protection against arbitrary physical compromise.

## 23. Takeaway (16:30-17:00)
End with the sentence the audience should remember. If time remains, point to the artifact and the
figures for reproducibility.

## 24. Backup: Useful Sources and Artifact Map (backup)
Keep this as a backup slide for Q&A or if someone asks where the implementation maps to the paper.

## Related-work source links used for the background slides

- Tramer et al., Stealing Machine Learning Models via Prediction APIs: https://arxiv.org/abs/1609.02943
- Orekondy et al., Knockoff Nets: https://arxiv.org/abs/1812.02766
- Fredrikson et al., model inversion attacks: https://dl.acm.org/doi/10.1145/2810103.2813677
- Lukas et al., SoK on DNN watermarking robustness: https://arxiv.org/abs/2108.04974
- Deep-Lock: https://arxiv.org/abs/2008.05966
- Prediction Poisoning: https://arxiv.org/abs/1906.10908
- AdvParams: https://arxiv.org/abs/2105.13697
- ActiveGuard: https://arxiv.org/abs/2103.01527
- GD-UAP: https://arxiv.org/abs/1801.08092