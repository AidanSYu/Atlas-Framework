# Atlas Roadmap

## Thesis

Every existing lab becomes an autonomous lab when the researcher's cognitive work is automated. Atlas is the brain. The bench stays the bench. The researcher proposes hypotheses, supervises, and does wetlab — Atlas handles literature, planning, optimization, recipe drafting, data parsing, retrosynthesis, ADMET filtering, novelty checking, run logging, and knowledge integration.

We are not building a greenfield robot SDL. We are not building a cloud lab. We are not building a workflow DSL. We are building the cognitive layer that retrofits onto any lab a PhD researcher already runs.

## The three goals, separated

| Property | Definition | Status |
|---|---|---|
| **Self-driving** | Atlas runs a campaign end-to-end once goal + constraints are set | Closest. Loop + plugins already exist. Need Campaign + BO + declarative envelope. |
| **Autonomous** | Minimal human intervention beyond goal-setting and wetlab actuation | Reframed for retrofit thesis — human IS the wetlab actuator. Need `actor` field + handoff pattern. |
| **Self-improving** | Each campaign makes the next campaign better. The lab compounds. | The moat. Substrate exists (SQLite + Qdrant + Rustworkx + trace JSONL); consumption loop does not. |

## Phases

### Phase 1 — Lock the harness shape (before any GPU time)

The training corpus shape is determined by what the orchestrator emits today. Get this right first.

- `actor` field on every tool call (`atlas` | `researcher` | `instrument` | `either`). `instrument` is unused at launch but reserved so the envelope does not change when labs add robots.
- Declarative tool envelope: `{tool, args, actor, preconditions, postconditions, expected_duration}`
- `Campaign` table in SQLite owning `[Task]`, carrying BO posterior + open-experiment list, resumable across process restarts
- `bayesian_optimize` core tool wrapping Atlas-BO library
- `requires_human_action` event in SSE stream; UI surfaces handoff; campaign pauses cleanly

### Phase 2 — Data ingest wedge

The way Atlas lands in an existing lab is by reading what the lab already produces. Not by replacing instruments.

- eLabFTW connector (read/write) — open source, academic-friendly, no API gatekeeping
- Three instrument file parsers: mzML, Bruker NMR FID, generic plate reader CSV
- One real bench tool: manual-protocol-runner UI (checklist-style) honest to the retrofit thesis. OpenTrons OT-2 optional for demo.

### Phase 3 — Self-improvement substrate (the moat)

- BO posterior carry-over keyed by problem class; new campaigns warm-start
- Negative results as first-class artifacts. New graph node type. UI surfaces "5 prior failed runs match this proposed setup."
- Lab-lore capture: detect recurring researcher overrides in trace log → propose `lab_lore` nodes → researcher confirms
- Drift heuristics: perplexity threshold + contradiction detection between tool calls within a run; surfaced inline

### Phase 4 — The 30B fine-tune

Only after Phases 1-3. The model is downstream of harness shape and trace shape.

- Trajectory curator: trace JSONL → SFT corpus, filtered for successful campaigns + corrected trajectories
- Dress rehearsal on 8B-14B first to validate corpus shape before burning H200-months
- Quarterly retraining cadence as trace volume grows — this is model-level self-improvement

### Phase 5 — Trust UX

- Hypothesis labels on every output with confidence + evidence trail
- Researcher override captured structurally as training signal
- One-click campaign export to PDF / PROV-O for publication, IP filing, regulatory

## What success looks like

### The two hero demos

**Retrofit demo.** A real chemistry lab running a 4-week campaign end-to-end on Atlas, with the researcher only doing wetlab + supervision. Measure: % cognitive work automated, # researcher overrides, time-to-result vs. their historical baseline. This is publishable. It is also the demo that closes academic sales.

**Compounding demo.** The same lab, second campaign on a related problem, with measurable improvement on iteration count, hit rate, or time-to-result attributable to what Atlas learned from campaign one. No SDL paper has published this. Owning it is the durable advantage.

### What "wow" looks like at launch

The launch demo is not a chatbot conversation about chemistry. It is a recorded screen of a real lab over multiple days:

- Day 1: PI gives Atlas a goal in two sentences. Atlas proposes 12 experiments, justifies each against retrieved literature + group prior runs, surfaces 3 expected dead ends from negative-result history.
- Day 4: researcher runs experiments 1-4 at the bench, uploads instrument files via drag-and-drop. Atlas parses them, updates the BO posterior, replans the remaining queue.
- Day 9: an experiment fails. Atlas attributes the failure to a likely cause from lab-lore ("LCMS column needed cleaning per recurring pattern from this researcher"), proposes a recovery experiment.
- Day 21: campaign closes. Atlas exports a publication-ready writeup with full provenance. The researcher edits, doesn't write.
- Day 22: a new campaign starts on a related target. Atlas warm-starts the BO from the prior posterior, surfaces the recipe templates that worked, skips the dead ends. The researcher hits a result in half the time.

That is the "this is a different category of software" moment. It is not LLM-as-chemistry-chatbot. It is the lab visibly compounding.

### Paying users

Initial paying segment: academic PIs running synthetic chemistry, medicinal chemistry, or formulation labs with 3-15 researchers, an existing ELN, and no robot. Price anchored to one postdoc-month per year per lab. The pitch: "it remembers what your postdocs forget, and the next campaign starts where the last one ended."

Distribution wedge: ELN connector + 4-week pilot with zero infrastructure change. Local install. No cloud dependency. IP-safe by default.

## The two-phase market

We sell software first and instruments never. But the architecture is built so labs buying new instruments later is a natural compounding move, not a re-platform.

**Phase A — software only (now through first generational result).** Atlas must be the best agentic lab framework in the world on existing hardware. The hero proof is a generational improvement — measurable acceleration on a real research target — delivered by software alone in a lab that bought nothing. Until we can demonstrate that, we do not ask any lab to spend capex.

**Phase B — robotics compounds the loop (after generational proof).** Once labs have seen Atlas pay for itself with software, the conversation becomes: "the bottleneck is now your hands. A $40k OT-2 plugged into Atlas removes it." The same plugin protocol, the same campaign loop, the same BO tool, the same handoff pattern — robotic actuators replace `actor: "researcher"` with `actor: "instrument"` on the steps where speed matters most. Atlas does not change shape. The lab gets faster.

This sequencing is deliberate. Selling robots first puts us in the greenfield-SDL market, competing on hardware integration we have no edge in. Selling software first puts us in a market with millions of researchers, no incumbent, and a flywheel that pulls hardware sales behind it later.

## Non-goals

These are explicit so we do not drift into the standard SDL field's traps.

- We are not building robot control *now*. The plugin protocol is shaped so robotic instruments slot in as a new actor type once labs choose to buy them. Robots compound speed; they are not a prerequisite, and we will not sell them.
- We are not building a workflow DSL. The researcher writes natural language; Atlas compiles to structured campaigns.
- We are not building cloud lab fulfillment. Samples stay in the researcher's lab.
- We are not building a chatbot UI. Single-turn Q&A is a degenerate case, not the product.
- We are not domain-specializing. The plugin protocol stays domain-agnostic; chemistry is the first wedge, not the limit.
- We are not optimizing the LLM for in-weights optimization. BO lives in a tool. The model orchestrates; it does not regress.

## The order that matters

Harness shape → data substrate → self-improvement loop → model fine-tune → trust UX. Reversing this order bakes a chatbot into the weights and we spend a year fighting it. Phase 1 is the highest-leverage week of work in the project.
