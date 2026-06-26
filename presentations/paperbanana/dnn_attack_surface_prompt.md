# Background Figure: Deployed DNN Attack Surface

This figure motivates why active protection is needed for deployed deep neural networks.

Scene:
- A high-value DNN model leaves the controlled cloud and is deployed inside edge or on-premise products, such as AI-enabled medical devices, hospital imaging systems, financial-service inference appliances, and industrial vision devices.
- In the cloud setting, access control, rate limits, logging, and server-side query monitoring can restrict abuse.
- In the edge or on-premise setting, the model is closer to the operator and adversary. Attackers can obtain practical oracle access through repeated low-latency queries, observed labels or probabilities, and sometimes physical access to deployed artifacts.

Attack surface around the deployed DNN:
- Model extraction: repeated queries and output probabilities are used to train a substitute model.
- Model inversion: confidence outputs are used to infer sensitive training information.
- Adversarial probing: adaptive queries reveal decision boundaries and model behavior.
- Model tampering or fine-tuning: attackers with deployed artifacts try to remove protection or recover utility.

Visual intent:
- Academic security diagram, clean and readable.
- Central element: "Deployed DNN on edge/on-premise device".
- Left side: legitimate deployment drivers: low latency, privacy, on-premise regulation, offline operation.
- Right side: attack arrows from an adversary to the deployed model, labeled with the four attacks above.
- Bottom: consequences: IP theft, privacy leakage, unsafe or unauthorized service, and loss of revenue/control.
- No solution mechanism should be shown; this is a motivation/background attack-surface figure.
