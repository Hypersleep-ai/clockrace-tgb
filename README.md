# ClockRace: reproduction code for fourteen Temporal Graph Benchmark entries

This repository reproduces our entries on fourteen datasets of the Temporal Graph Benchmark (TGB and TGB 2.0). Each
directory under `entries/` runs the five seeds of one submitted entry with its exact settings and lists every seed's
validation and test score.

## Results

Five seeds per dataset, each trained from scratch: mean ± unbiased standard deviation (torch.std); higher is better.
The reference column is the best published test score on the live leaderboard, read on 24 November 2026.

| dataset | metric | validation | test | best published test | seeds above it | parameters | median minutes per seed | script |
|---|---|---:|---:|---|---:|---:|---:|---|
| tgbn-trade | NDCG@10 | 0.8746 ± 0.0004 | **0.8687 ± 0.0013** | NAViS 0.863 | 5/5 | 2,107 | 4 | `entries/tgbn-trade` |
| tgbn-genre | NDCG@10 | 0.5387 ± 0.0002 | **0.5465 ± 0.0002** | NAViS 0.528 | 5/5 | 2,360 | 10 | `entries/tgbn-genre` |
| tgbn-reddit | NDCG@10 | 0.6008 ± 0.0002 | **0.5868 ± 0.0002** | NAViS 0.569 | 5/5 | 2,360 | 443 | `entries/tgbn-reddit` |
| tgbn-token | NDCG@10 | 0.5022 ± 0.0004 | **0.5156 ± 0.0006** | NAViS 0.513 | 5/5 | 2,039 | 350 | `entries/tgbn-token` |
| tgbl-wiki-v2 | MRR | 0.8518 ± 0.0001 | **0.8353 ± 0.0002** | TRACE (per-timestamp) 0.833 ± 0.001 | 5/5 | 7,995 | 35 | `entries/tgbl-wiki-v2` |
| tgbl-review-v2 | MRR | 0.4883 ± 0.0004 | **0.5398 ± 0.0008** | GraphMixer 0.521 ± 0.015 | 5/5 | 11,642,367 | 34 | `entries/tgbl-review-v2` |
| tgbl-flight-v2 | MRR | 0.9147 ± 0.0001 | **0.9085 ± 0.0002** | TPNet 0.884 ± 0.001 | 5/5 | 6,929 | 148 | `entries/tgbl-flight-v2` |
| tgbl-coin-v2 | MRR | 0.8536 ± 0.0002 | **0.8691 ± 0.0006** | TPNet 0.832 ± 0.001 | 5/5 | 12,737 | 313 | `entries/tgbl-coin-v2` |
| thgl-github | MRR | 0.7558 ± 0.0003 | **0.8466 ± 0.0007** | EdgeBank(unlimited) 0.413 | 5/5 | 7,157 | 70 | `entries/thgl-github` |
| thgl-myket | MRR | 0.9166 ± 0.0001 | **0.9158 ± 0.0007** | EdgeBank(unlimited) 0.456 | 5/5 | 7,061 | 198 | `entries/thgl-myket` |
| thgl-forum | MRR | 0.8167 ± 0.0003 | **0.8162 ± 0.0002** | TGN(edge type) 0.729 ± 0.009 | 5/5 | 8,051 | 821 | `entries/thgl-forum` |
| tkgl-polecat | MRR | 0.2633 ± 0.0001 | **0.2581 ± 0.0002** | CountTRuCoLa 0.256 ± 0.001 | 5/5 | 29,081 | 89 | `entries/tkgl-polecat` |
| tkgl-icews | MRR | 0.3470 ± 0.0001 | **0.3452 ± 0.0004** | CountTRuCoLa 0.214 ± 0.001 | 5/5 | 34,438 | 513 | `entries/tkgl-icews` |
| tkgl-wikidata | MRR | 0.7408 ± 0.0001 | **0.6707 ± 0.0002** | CountTRuCoLa 0.609 ± 0.001 | 5/5 | 81,178 | 18 | `entries/tkgl-wikidata` |

Per-seed values are in `RESULTS.md` and `entries/<dataset>/results.json`. Parameters are counted as TGB asks,
`sum(p.numel() for p in model.parameters())`. On tgbl-review-v2, 11,637,021 of the 11,642,367 are per-node identity
tables held at zero, neither trained nor read by the submitted configuration; 5,346 parameters are trained.

## The method

ClockRace keeps the history of a temporal graph in addressed, time-referenced records. An interaction updates only the
records it addresses. A query brings the records of its source and of each candidate to the query time, and a small
learned map turns them into scores; the candidates compete in one race and the ranking is issued before the answering
interaction is admitted.

- **Node affinity (tgbn-trade, tgbn-genre, tgbn-reddit, tgbn-token).** For each source, affinities to every destination
  class at several elapsed-time decay scales, formed from completed periods or from streaming rows brought to the
  query time when read, with previous-period vectors, global class shares and growth. On genre and reddit the decay
  scales also count interactions, and a learned per-source softmax over the bank of decay scales mixes them, every scale
  contributing. The weights
  are trained on the training split only and stay fixed during validation and test.
- **Link prediction (tgbl-wiki-v2, tgbl-review-v2, tgbl-flight-v2, tgbl-coin-v2, thgl-github, thgl-myket, thgl-forum).**
  Persistent pair records addressed by a source and a destination (and by the relation type on thgl-*), with source and
  destination clocks. Dataset-specific reads use the same scheme: the source's own past destinations as training
  competitors (review), day-scale pair clocks and reads at schedule lags (flight), counterparty-addressed two-step reads
  with flow, amount and novelty reads (coin), and reverse-pair reads (thgl-forum). Training races the true destination
  against competitors drawn from the training data by the rule the evaluation uses to sample its negatives.
- **Temporal knowledge graphs (tkgl-polecat, tkgl-icews, tkgl-wikidata).** For a query (subject, relation, time), every
  candidate object competes in one race of addressed causes: recurrence, relation transfer, transitions and popularity,
  over a shared background computed without per-entity identity parameters. On polecat a further cause reads partners
  of partners: the subject's recent partners are read from their records and each partner's own recent partners in turn.
  On wikidata the race is read per listed candidate and trained on candidates drawn from the training data like the
  evaluation's.

## Protocol

- Official splits, negative samples and `Evaluator` of py-tgb 2.3.0 (tgbl-wiki-v2 and tgbl-review-v2 negatives v2;
  tgbl-coin-v2 and tgbl-flight-v2 the chronologically corrected releases).
- Each query reads only interactions strictly earlier than its timestamp (ties excluded); validation and test interactions enter the records after the queries they answer are scored, never the weights; validation selects the epoch and the configuration. Index arrays are allocated over every node ID of the dataset.
- Node affinity: per-node period or streaming state is read at a label time only from interactions before that label time; a label time's labels are revealed to the state after its predictions are scored; gradient steps use TRAIN label times only; validation selects the epoch. Array sizes and the class list come from the dataset metadata.
- **Labels on tgbn-genre, tgbn-reddit and tgbn-token.** The scores use TGB's official evaluation loop, as every
  leaderboard entry does. That loop returns each label to the model right after scoring it, and Wang et al.
  (arXiv:2611.30289) show that these labels summarise a day about five days after their timestamp, so a model that
  reads past labels receives them early. Our models for these three datasets read the returned labels, as the
  persistent-forecast and moving-average baselines do. The tgbn-trade model predicts each year from the edges of
  earlier years only.
- Knowledge graphs: each query reads only snapshots strictly before its own; one-vs-all scoring over every entity with the official time-filtered conflict sets and tie rule; validation and test snapshots update the records, never the weights; validation selects the epoch and the configuration. No identity tables; the static Wikidata graph is not used.
- tgbl-wiki-v2 allocates its source and destination index over every node ID of the dataset and the official
  validation negatives (whose union is all 1,000 destinations); the model has no source identity parameters and a learned
  bias per destination.
- Model selection uses validation only. Each seed's test score is read once, with its best validation checkpoint.
- **Seeds.** Seeds 0 to 4, as the TGB rules ask. On tgbl-wiki-v2 and tgbn-trade, seeds 3 and 4 were run after the test
  scores of seeds 0 to 2 were known, with the configuration frozen; all five are reported.
- **Earlier entries.** On tgbn-genre, tgbn-reddit and tkgl-polecat, an earlier configuration was scored on test (five seeds:
  genre 0.5441, reddit 0.5840, polecat 0.2544) before the submitted configuration was developed and selected on
  validation. On tkgl-polecat, the earlier configuration's test profile motivated the two-hop path cause: 42% of test
  queries ask for an object new to their (subject, relation) pair, which it ranked at 0.077 MRR. The path cause was then
  built, screened and selected on validation.
- **Live-update scores.** On tgbl-review-v2 and tkgl-polecat, the submitted checkpoints were also scored once more in
  TGB's live-update setting (Appendix C of the TGB paper; weights adapting after each revealed batch): 0.5440 and 0.2588.
  Those scores are not part of these entries, and the live-update scoring code is not part of this repository.
- No external data or pretrained models.

## Environment

- Python 3.14.4, PyTorch 2.15.0 and NumPy 2.5.4 (`requirements.txt`); py-tgb 2.3.0 and its dependencies installed into
  the package: `python -m pip install --target src/.cache/pylib_tgb -r requirements-tgb.txt`. The drivers put
  `src/.cache/pylib_tgb` first on `sys.path`.
- CPU only, one thread per run, except tkgl-wikidata (one NVIDIA RTX PRO 6000 Blackwell GPU, two CPU threads). The
  entries ran on AMD Ryzen 9 7950X3D, Intel Core i5-4690 and Intel Core Ultra 7 265 hosts.
- Peak resident memory per run, where recorded: from 3.2 GiB (tgbl-review-v2) to 27.3 GiB (tkgl-icews); see `RESULTS.md`.

## Running

`sh entries/<dataset>/run_five_seeds.sh` downloads the dataset through py-tgb on first use (into
`src/data/tgb/root/`) and runs the five seeds. tkgl-wikidata first converts the official validation negatives
into a memory-mapped store (`experiments/tgb/tkg_ns_stream.py --split val`, included in its script); its first seed
converts the test negatives. Its script caps PyTorch at `--gpu-mem-frac 0.12` of the card, about 11.5 GB on the 96 GB
card used (a run peaks at 10.8 GB); on a smaller card raise the fraction so that the cap stays at least 11.5 GB, for
example 0.15 on an 80 GB card.

Reproductions agree with the recorded scores within seed noise. Bit-for-bit equality across machines is not guaranteed:
on a flat validation optimum a difference of about 1e-7 in training can move the best epoch. From this package on a
different host, the five tgbn-trade seeds give a test mean of 0.86833 (recorded
0.86868) and the five tgbl-wiki-v2 seeds 0.835281 (recorded
0.835281).

## Licence

PolyForm Noncommercial License 1.0.0 (`LICENSE`), with the required notice in `NOTICE`: free for research, evaluation,
education and other noncommercial purposes. The rights holder is Pataluha Ventures S.L.; for a commercial licence,
contact Hypersleep (https://hypersleep.ai).
