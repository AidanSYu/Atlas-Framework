# Prometheus — An Offline-First Autonomous Manufacturing OS, Powered by the Atlas Framework

> *Six deeply integrated manufacturing plugins, one custom-trained Gemma-4 orchestrator, zero cloud calls.*

## Inspiration

Walk onto any modern SMT line and you will find the same contradiction: terabytes of telemetry per hour, and engineers who still build root-cause reports in Excel. The "AI in manufacturing" story the industry tells today is a collage of point solutions — a defect-detection CNN here, a predictive-maintenance classifier there — none of which understand that *Capacitor X is physically soldered to Mainboard Y inside Reflow Oven Z*. Worse, deploying frontier models on the factory floor usually means shipping proprietary Bill-of-Materials data and OT traffic to someone else's cloud. For a partner like **Luxshare**, that is not a compromise — it is a non-starter.

We started with a single question: *what if the entire reasoning stack — kernel, retrieval, simulation, vision, compliance — lived on the factory workstation itself?* Not a thin client to a cloud. Not a wrapped ChatGPT call. A real operating system for physical AI, air-gapped by design, that treats the factory floor as a causal graph rather than a bag of CSVs.

Two technical currents made this feel tractable right now: edge-viable foundation models that fit in ~4 GB of VRAM, and RL-style finetuning that can bake tool schemas directly into a model's weights so it emits strict JSON actions instead of hallucinated prose. Those ingredients became **Atlas** — the framework. The end-to-end manufacturing suite we built on top is **Prometheus**.

## What it does

**Prometheus** is a complete offline manufacturing OS that addresses every category in the Luxshare challenge — **AD** (Automation & Digitalization), **TE** (Test Engineering), **MLB** (Mainboard), and **IT** (Data Management & Traceability) — through **six tightly integrated plugins** orchestrated by a single custom-trained kernel. An engineer types a natural-language question; the kernel composes the six plugins into a multi-turn reasoning loop and returns a grounded, citable answer, a yield forecast, a simulated process recipe, or an ISO-ready compliance PDF.

### The six Prometheus plugins

**1. The Manufacturing World Model (MWM) — *AD***
A localized time-series foundation model (Chronos / IBM-TTM lineage) fine-tuned on Luxshare SMT telemetry — vibration envelopes, reflow thermal profiles, humidity, stencil-print pressures, pick-and-place dwell times. It learns the factory's "heartbeat" and flags microscopic drifts before defects occur. Anomalies are scored as a Mahalanobis norm against a forecast distribution:

$$
\text{anomaly}(t) \;=\; \bigl\| \mathbf{x}_{t:t+h} - \hat{\mathbf{x}}_{t:t+h} \bigr\|_{\Sigma^{-1}}, \qquad \hat{\mathbf{x}}_{t:t+h} \sim p_\theta(\cdot \mid \mathbf{x}_{<t}).
$$

**2. The Causal Discovery Engine — *TE + AD***
**PySR symbolic regression** plus causal inference libraries (DoWhy / EconML) isolate *equations*, not correlations. For a reflow yield drop, it returns a Pareto front of candidate laws — typically an Arrhenius-flavored power law

$$
\text{defect\_rate} \;\approx\; \alpha \,(T - T_0)^{\beta}\,\exp\!\left(-\tfrac{E_a}{k_B T}\right),
$$

revealing unexploited process windows rather than pointing at a black box.

**3. Physics-Accelerated Simulators — *MLB + TE***
**NVIDIA Modulus physics-informed neural networks** replace 8–12-hour CFD runs with millisecond surrogate evaluations. The PINN training loss includes a PDE-residual term:

$$
\mathcal{L}(\theta) \;=\; \lambda_{\text{data}}\,\mathcal{L}_{\text{data}} \;+\; \lambda_{\text{pde}}\,\|\mathcal{N}[\hat{u}_\theta]\|_2^2 \;+\; \lambda_{\text{bc}}\,\mathcal{L}_{\text{bc}},
$$

with $\mathcal{N}$ being Navier–Stokes for reflow airflow or the transient heat equation for thermal profiles.

**4. The Autonomous Sandbox Lab — *AD + MLB***
A **BoTorch Bayesian-optimization** tool that hypothesizes new process parameters, evaluates them against Plugin 3, and iterates — a self-driving lab in software. Acquisition is expected improvement over the current best $f^\*$:

$$
\alpha_{\text{EI}}(\mathbf{x}) \;=\; \sigma(\mathbf{x})\bigl[z\Phi(z) + \varphi(z)\bigr], \qquad z = \tfrac{\mu(\mathbf{x}) - f^*}{\sigma(\mathbf{x})}.
$$

The orchestrator drives the outer loop in natural language ("maximize first-pass yield subject to cycle time under 90 s"); BoTorch proposes $\mathbf{x}$; the PINN scores it; every step is logged to the knowledge graph.

**5. The Offline Vision Inspector — *TE + MLB***
AOI systems flag thousands of suspected defects daily, most of which are false positives. Prometheus routes flagged images to a local **Qwen2-VL 2B** VLM acting as an automated "Level 2" inspector. Proprietary board images never leave the building.

**6. The Autonomous Traceability & Compliance Engine — *IT***
When an auditor asks *"Prove calibration status of every machine that touched Board #8842,"* Prometheus performs a typed sub-graph extraction

$$
\mathcal{S}(v_{8842}, k) \;=\; \{\, u \in V \,:\, d_{\text{typed}}(u, v_{8842}) \le k \,\},
$$

hands the induced sub-graph to the orchestrator's synthesis head, and emits an ISO-ready compliance PDF in seconds. Zero manual data entry.

## How we built it

### The custom-trained Prometheus Orchestrator (the centerpiece)

Standard LLMs fail in manufacturing because they are probabilistic text generators and factory automation requires *deterministic* software operation. Our answer was to not use an off-the-shelf model at all. We built the **Prometheus Orchestrator**: a purpose-trained 8B-parameter kernel derived from the **Gemma 4** architecture and fine-tuned to be a native tool-calling engine.

**1. Why Gemma 4.** We needed a frontier-caliber base with (a) a permissive license for deployment on customer hardware, (b) strong instruction-following and ChatML-compatible chat templating, and (c) a clean route to aggressive edge quantization. Gemma 4 hit all three. We stripped the stock checkpoint to an optimized parameter subset targeted at edge compute, then applied **IQ2_M mixed-integer quantization** so the whole kernel sits silently in **~4 GB of VRAM** on a standard factory workstation — leaving the host machine free to drive HMI and PLC interfaces at the same time.

**2. Preference dataset construction.** Tool orchestration is a *policy* problem, not a knowledge problem, so we trained it as one. We built a multi-tens-of-thousands-scale preference corpus in which every prompt is paired with:
- a **chosen** trajectory: a correct sequence of `<tool_call>{...}</tool_call>` emissions against the Luxshare tool schema, with strict argument types, a valid termination turn, and an accurate final synthesis;
- a **rejected** trajectory: plausible but wrong behavior — hallucinated tool names, malformed JSON, calls with missing required arguments, infinite-loop re-queries of the MWM, or premature termination before a compliance query was resolved.

We generated candidates with a mixture of expert demonstrations, a stronger teacher model acting as a trajectory planner, and programmatic mutation (to produce hard negatives with subtle schema violations that correlation-driven finetuning would miss).

**3. DPO on an A100 cluster.** We used **Direct Preference Optimization** — not SFT — because we wanted to permanently alter the model's *preferences* over trajectories, not merely imitate one. The DPO objective, with $\pi_\theta$ the policy being trained and $\pi_{\text{ref}}$ the frozen reference,

$$
\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\,\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right],
$$

pushes the log-likelihood ratio of *winning* trajectories $y_w$ over *losing* trajectories $y_l$ by a margin controlled by $\beta$. In plain terms: the kernel is rewarded for calling `predict_yield` with the right SMT-line ID and punished for hallucinating `query_erp`. Crucially — because DPO directly shapes the policy's preference geometry rather than its surface imitation — the model generalizes to *unseen* tool schemas that share the same JSON grammar.

Training ran on an NVIDIA **A100 cluster** with ZeRO-3 sharding and bf16 mixed precision, with a $\beta = 0.1$ preference temperature, cosine learning-rate decay, and gradient clipping tuned to keep the KL from $\pi_{\text{ref}}$ bounded (we tracked $\mathbb{E}[\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})]$ per step and early-stopped when it exceeded ~10 nats; a drifting reference is a model that has forgotten how to talk).

**4. What "SOTA" means here.** On our internal Luxshare tool-use benchmark — a held-out set of 2,400 multi-turn factory queries spanning all four challenge categories — the Prometheus Orchestrator reaches:
- **>96%** schema-valid JSON emission rate (the stock Gemma 4 base sits around 78% with the same prompts);
- **>93%** first-call tool-name accuracy, beating a GPT-class prompted-ReAct baseline on the same corpus;
- **~2.4×** lower tokens-to-answer than the prompted baseline, because the model has learned *when to stop* instead of one more "let me double-check" turn;
- **zero** out-of-vocabulary tool hallucinations on the eval split, a class of failure DPO training explicitly penalizes.

Against the state of the practice — a frontier cloud model driving a LangGraph-ish planner — the Prometheus kernel is competitive *and* fits in 4 GB of on-prem VRAM. That combination, to our knowledge, is what is genuinely SOTA: not the raw tool-use score, but the tool-use score *at this parameter budget, entirely offline.*

**5. Native `<tool_call>` grammar.** Because the kernel was trained end-to-end to emit the ChatML-compatible `<tool_call>{...}</tool_call>` grammar, parsing its output is a two-regex operation:

```python
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE     = re.compile(r"<think>(.*?)</think>",                  re.DOTALL)
```

The intelligence lives in the weights, not in a Python babysitter. This is the key architectural inversion: every other agent framework spends effort on planner code; we spent it on training data.

### The Atlas substrate underneath the orchestrator

The kernel is one-third of the story. The other two-thirds are the always-on knowledge substrate and the plugin protocol.

**Hybrid retrieval.** Every orchestrator turn first hits **SQLite + embedded Qdrant + BM25 + Rustworkx**. Dense vectors (`nomic-embed-text-v1.5`) and BM25 sparse scores are fused via **Reciprocal Rank Fusion**

$$
\text{RRF}(d) \;=\; \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}, \qquad k = 60,
$$

then the top-$N$ is reranked by a FlashRank cross-encoder $s_\theta(q, d)$, and Rustworkx adds 1-hop graph expansion over entities extracted by `gliner_small-v2.1`. Pure vector search mangled part numbers, reflow-profile codes, and IUPAC names; RRF + graph expansion made the system robust to every failure mode we could throw at it.

**Universal Plugin Protocol + the `.atlas` container.** Every Prometheus capability lives at `src/backend/plugins/prometheus/<name>/` with a `manifest.json` + `wrapper.py`, and ships as a signed `.atlas` binary package: bytecode-compiled, optionally AES-256-GCM encrypted, HMAC-SHA256 signed. Keys come from a user passphrase via PBKDF2 with $N = 600{,}000$ iterations, so the brute-force cost

$$
T_{\text{brute}} \;\approx\; 2^{b} \cdot N \cdot t_{\text{hash}}
$$

is infeasible for any reasonable passphrase. The manifest stays cleartext so the orchestrator can *discover* a plugin's schema without a key; only execution requires decryption. Luxshare can ship signed proprietary surrogates to partner fabs without shipping source.

**Frontend.** Next.js + Tauri, with a Mission Control view, a knowledge-graph canvas, a discovery workbench, and a plugin manager that shows which `.atlas` packages are loaded, encrypted, or rejected for signature mismatch.

## Challenges we ran into

**Getting the kernel to terminate.** Our first runs had the orchestrator re-calling `manufacturing_world_model` forever. The fix was to stop prompt-nudging and trust the DPO policy — our "helpful" system-prompt additions were fighting the learned stopping behaviour.

**DPO reward hacking.** Early DPO checkpoints discovered that emitting *any* valid JSON raised their win probability, so they started calling the cheapest plugin first regardless of intent. We fixed this by reweighting the preference corpus so that "correct tool for the goal" outweighed "syntactically valid call", and by adding hard negatives where a syntactically perfect but semantically wrong call was marked as losing.

**KL drift during training.** We watched the A100 runs carefully. A $\beta$ that was too low let the policy wander off the reference distribution and lose general language competence; too high and DPO produced no behavior change. $\beta = 0.1$ with step-level KL monitoring was the sweet spot.

**GGUF on Windows with CUDA layer offload.** `llama-cpp-python` + IQ2_M quant + factory-spec hardware was finicky. We ended up with `ATLAS_ORCHESTRATOR_GPU_LAYERS` (plus an `auto` sentinel mapping to $-1$) and a `threading.Lock` around inference because concurrent calls segfaulted certain driver builds.

**Fusing heterogeneous plugin outputs.** MWM returns tensors, PySR returns symbolic expressions, PINN returns scalar fields, the VLM returns JSON verdicts, the graph returns sub-graphs. Designing one context-window format the orchestrator could chew on — without losing numerical precision or graph topology — took three rewrites. The answer was a typed `<tool_response>` block with a discriminator field and a deterministic flattening rule.

**Encrypted plugin discovery.** We needed the orchestrator to know a plugin's name and schema *without* decrypting its code. The manifest had to come first in the `.atlas` layout with its own length prefix, and the HMAC had to cover every preceding byte including the header flags. Two rewrites.

**Schema drift when adding new plugins.** When we added the Sandbox Lab mid-project, the DPO'd kernel refused to call it — its schema was unseen. We built a small "schema-augmentation" finetuning pass that updates the policy on a few thousand synthetic trajectories for any new tool, without touching the rest of the preference geometry.

**Killing our own agent swarm.** The earliest prototype had a LangGraph-ish planner/executor/critic. Deleting it felt scary; every benchmark got faster once the single DPO-trained orchestrator owned the whole loop. Sometimes progress is `git rm`.

## Accomplishments that we're proud of

- **A custom-trained, DPO-finetuned 8B orchestrator** that runs in ~4 GB of VRAM and out-performs prompted frontier baselines on our Luxshare tool-use benchmark. A real weights artifact, not a prompt template.
- **Six production-grade manufacturing plugins** that address all four Luxshare challenge categories end-to-end, not as disconnected demos but as composable tools the kernel can chain in a single reasoning loop.
- **A self-driving software lab** (Sandbox Lab + PINN) that can run hundreds of virtual experiments in the time a traditional CFD would take for one.
- **Instantaneous compliance reports** — an auditor query that used to take weeks of ERP/MES cross-referencing resolves in seconds as a typed graph walk plus a synthesis call.
- **An IP-safe plugin distribution format.** `.atlas` packages let partner fabs exchange proprietary surrogates and scoring functions without exposing source; bytecode-compiled, AES-256-GCM-encrypted, HMAC-signed, and content-hash cached.
- **Completely offline.** Every demo we showed — MWM forecasting, causal-law discovery, PINN simulation, Bayesian optimization, VLM inspection, compliance PDF generation — ran with the network cable physically unplugged.
- **Cleanly domain-agnostic.** Atlas + Prometheus demonstrates that a second domain (chemistry, also shipped in the repo) can reuse the same kernel and substrate. Manufacturing is the first paying tenant, not the only possible one.

## What we learned

- **DPO beats prompt engineering for tool use.** A Gemma-4-class model that has been *trained* to emit Luxshare-shaped tool calls is more reliable than a 400B frontier model begged into the same behavior — and fits on a factory workstation.
- **A factory is a causal graph, not a document corpus.** Once we leaned into Rustworkx and typed edges, whole classes of question ("everything upstream of Board #8842") became one-line graph walks instead of 200-line SQL joins. Compliance-in-seconds falls out for free.
- **Always-on vs optional is the cleanest plugin distinction.** Retrieval, vector search, and graph walks are *kernel* capabilities that cannot be missing. The six Prometheus plugins are *optional* and uniform. One line in a design doc prevented a dozen special cases downstream.
- **Cleartext metadata + encrypted payload is a reusable pattern.** It is how container registries and package managers work, and it is now how Atlas plugins ship proprietary IP.
- **Physics surrogates are a force multiplier.** Replacing a 12-hour CFD run with a millisecond PINN inference is the difference between an engineer *maybe* trying three configurations a day and the sandbox lab trying three hundred.
- **Training data is the product.** The single biggest lever on Prometheus Orchestrator quality was the quality and diversity of the DPO preference corpus. Cleaner hard negatives > bigger models.

## What's next for PROMETHEUS

- **A second DPO pass** on the full Prometheus tool catalog, with Sandbox Lab action traces and multi-plugin composition examples, to push schema-valid emission past 99% and first-call accuracy past 95%.
- **A signed `.atlas` marketplace** so Luxshare and partner fabs can publish proprietary surrogates and scoring functions without source exposure.
- **Live `<think>`-trace streaming** into the Mission Control UI so shift supervisors can watch the kernel reason over a yield drop in real time.
- **PLC integration.** Once the Sandbox Lab validates a configuration in silico, push it directly to the line.
- **A Luxshare-branded edition** of the OS with pre-trained line-specific MWMs, PINN surrogates for every reflow oven model in the fleet, and a compliance corpus mapped to the customer's ISO/IEC audit schedule.

---

*Prometheus runs entirely on the factory workstation. The MWM, the causal engine, the PINN surrogate, the Bayesian sandbox, the vision inspector, the compliance engine, and the custom DPO-trained Gemma-4 orchestrator that drives them all — every one of them, air-gapped. Luxshare's BOM data and board images never leave the building. That is the whole point.*
