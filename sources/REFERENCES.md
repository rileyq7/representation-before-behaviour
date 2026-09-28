# Sources and provenance

- Patel, Sivakumar, Theobald, Zappella, Apostoloff. **Fairness Dynamics During Training.** NeurIPS EvalEval workshop 2024; arXiv:2506.01709 (2025). https://arxiv.org/html/2506.01709v1
- Apple research page: https://machinelearning.apple.com/research/fairness-dynamics
- Biderman et al. **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.** ICML 2023. Official code/model documentation: https://github.com/EleutherAI/pythia ; local README snapshot `pythia-README.md`.
- Zhao et al. **Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods.** NAACL 2018. WinoBias original repository: https://github.com/uclanlp/corefBias ; downloaded Type 2 data and occupation labels are listed with SHA-256 hashes in `data-manifest.json`. Repository data retain their original ownership and terms.
- Official model: https://huggingface.co/EleutherAI/pythia-1.4b . Each downloaded revision is resolved to an immutable commit and saved in checkpoint provenance.

The Apple prompt figure was inspected visually (`apple-prompts.png`). The paper describes option randomisation with five seeds but does not provide the exact seeds in its text. This project's ASCII-punctuation prompts and deterministic seeds are reconstructed and explicitly labelled as such.

Main correction to the initial project premise: the reported ~80k divergence belongs to **Pythia-6.9B**, not Pythia-1.4B. The publication also already uses continuous output probabilities. Its overall male/female performance gap and occupation-specific stereotype preference must not be treated as identical quantities.

Source retrieval date: 2026-09-21. The Firecrawl text cache is under `.firecrawl/`; the public original sources above are authoritative.
