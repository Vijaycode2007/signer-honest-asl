# Signer-honest sign language recognition

A vision backend that reads a still photo of an ASL fingerspelling handshape and returns a letter, served over HTTP behind a web app that builds a message one photo at a time.

The models are ordinary.
What is not ordinary is the evaluation, and that is the point of the project.

Every number below was measured by code in this repository, on splits this repository constructs and audits.
Where a number is a judgement call rather than a measurement, it says so.

## The result

Protocol C is the honest one: train on one corpus, test on a different corpus shot by different people in a different room.

| pipeline | accuracy | macro-F1 | log-loss |
| --- | --- | --- | --- |
| pixels only (DeiT III base, 85.8M params) | 0.7034 | 0.6809 | 1.2242 |
| hand geometry only (MLP, 90K params) | 0.9034 | 0.8965 | 0.6985 |
| **fused, 7.5% pixels + 92.5% geometry** | **0.9080** | **0.9005** | **0.4263** |

A 90,000-parameter MLP over 21 hand landmarks beats an 85.8-million-parameter vision transformer by 20 accuracy points on the only protocol that predicts what happens on a stranger's phone.

That is not a claim about model architecture.
It is the thesis of this repository arriving in a form nobody can argue with: the pixel model was never that good, it was reading the room.
Landmarks have nowhere to store a wallpaper, so nothing transfers except the shape of the hand.

## The problem this project is actually about

Published static-ASL results cluster around 99.9%.
A widely cited 2023 Vision Transformer paper reports 99.98% on the ASL alphabet.
Those numbers do not survive contact with a phone camera, and the reason is not the architecture.

The standard corpora are continuous webcam captures.
`asl_alphabet`, the most-used Kaggle set, is 87,000 frames of essentially one person in one room.
Split those frames at random and frame 1041 lands in train while frame 1042 lands in test.
The two images differ by a few pixels of hand tremor.
A classifier that memorises the wallpaper scores a perfect mark.

Sign Language MNIST is worse: its official test split is produced by augmenting the same base photographs as its train split.
It is leaky by construction, and it is still used as a benchmark.

So this repository reports three numbers for the same model on the same data:

| protocol | what it measures | measured |
| --- | --- | --- |
| **A** random split | what everyone else publishes | 1.0000 accuracy, 99.14% of the test set had a near-duplicate in train |
| **B** group split | capture sessions kept whole | 0.6318 |
| **C** cross-corpus | tested on a corpus shot by different people | 0.6057 |

Those three are the ViT-Small baseline, the one model that has been through all three.
A minus C is the contribution: **39 points of the published headline was leakage, not skill.**

## What is here that other projects do not have

**A leakage audit that produces a number.**
`slr.leakage.audit` hashes every image and reports what fraction of the test split has a near-duplicate sitting in train.
Run it on any split, including someone else's, and the honesty of that split stops being a matter of opinion.
Banded hashing with a pigeonhole guarantee makes it exact and fast enough for 90k images.

It is not decorative.
Pointed at `asl_alphabet_test`, the holdout corpus, it found that **51.7% of a random half of it has a near-duplicate in the other half**, 406 of them pixel-identical.
Any procedure that splits that corpus at random is leaking, including one that only wanted a small tuning slice.

**Group recovery for corpora that ship no signer labels, and an honest account of its limits.**
For `Sign-Language-Digits-Dataset`, signer identity is recoverable exactly: 218 students were photographed back to back on one camera, ten shots each, digits 0-9 in order, so the camera's frame counter encodes the student (`signer = (IMG_number - 1118) // 10`).
Verified on the real download: 2062 images resolve into 217 signers.

For everything else, `recover_groups` clusters near-duplicates into connected components.
That recovers **capture sessions**, not signers, and the distinction is the difference between a real claim and a marketing one.
Scored against the digits corpus's ground-truth signer ids, every signer fragments at any usable threshold:

| max_dist | components | signers fragmented across components |
| --- | --- | --- |
| 2 | 2021 | 217 / 217 |
| 6 | 953 | 211 / 217 |
| 10 | 141 | 88 / 217 |

That limitation is exactly why the headline metric is cross-corpus rather than an in-corpus group split.

**Hand localisation, with the detection rate reported rather than assumed.**
MediaPipe finds the hand, OpenCV does the geometry.
`slr.hands` reports what it actually achieves over every image rather than a sample: 71,717 of 87,000 on `asl_alphabet` (82.4%) and 1,658 of 1,740 on the holdout (95.3%).

Much of the shortfall is not failure.
3,000 of the 15,283 misses are the `nothing` class, which is an empty room with no hand in it.
Looking twice earns most of the rest: a padded, upscaled retry on a miss took a 300-image sample from 74.7% to 81.3%.

Contrast normalisation (CLAHE) was tried and **lowered** detection to 64%.
It is not in the pipeline, and that sentence is in the source so nobody adds it back assuming it must help.

**Chirality folding, which is the difference between working and not working for half the population.**
`asl_alphabet` is 92.5% right hands and the holdout corpus 98.3%.
The two corpora agree with each other and disagree with everyone who signs left-handed, so no benchmark in this field would ever expose the problem.

On 12 photos from a left-handed signer the geometry model scored **2/12**.
Folding left hands onto the right-handed frame before building features took it to **9/12**, with no retraining.
Handedness comes from MediaPipe per image, so nothing has to be configured and the API takes no flag.

This is deliberately not flip augmentation, which `slr.data` refuses for pixels because mirroring an image can change what a sign means.
Nothing is taught that a mirrored sign is the same sign.
The reflection is undone before features are built, so the model only ever sees one chirality.

**Calibration and abstention, because the app needs them.**
A confident wrong letter is worse than "I'm not sure".
`slr.evaluate` fits a temperature on validation and produces a risk-coverage table, and the service reads its threshold from that table.

**An architecture search that cannot cheat.**
Eighteen backbones are registered.
Candidates are ranked on **validation** macro-F1 and only the winner is ever run against the test split.
Two plain CNNs are in the search as controls rather than filler.

**Attention rollout as evidence.**
`POST /api/predict?explain=true` returns the heatmap overlaid on the photo, which makes a model attending to the wall behind the signer visible immediately.

## The fused pipeline

Two models vote on the same 29 classes, blended as a linear probability pool at **7.5% pixels / 92.5% geometry**.

The ratio was not chosen by taste, and choosing it correctly turned out to be the hard part.

The holdout corpus is 51.7% internally near-duplicated, so a random half-split leaks.
Recovering near-duplicate groups first and splitting on those gives two halves with **0% leakage** between them.
The weight was swept in 0.025 steps on one half and reported once on the other, and neither model was trained on either.

Read the log-loss column rather than the accuracy column.
Fusion buys half a point of accuracy over geometry alone and cuts log-loss by 39%, and log-loss is the half that matters here, because the abstention threshold is a cut on a confidence and a better-calibrated confidence is a better-placed refusal.

**The instructive failure:** selecting that same ratio on the in-corpus validation split gives a weight of **zero**, which is pixels only, the worst of the three.
Within-corpus validation rewards whichever model best exploits the corpus.
That is the exact bias this project exists to measure, and it will happily eat a fusion weight if you let it.

The geometry model is an MLP over the 21 landmark coordinates, wrist-centred, palm-scaled and folded right-handed.
Explicit per-finger features (joint angles, fingertip-to-palm distances, tip spread) were built and measured, and they lose: fused, they score 0.8460 against 0.9080.
`slr.landmarks.finger_features` is kept because it is legible, not because it predicts.

## Honest limits

**The three protocol numbers are not all the same model.**
A, B and C above are ViT-Small.
The served DeiT III scores 0.6977 on protocol C.
Any comparison across those rows is a comparison of protocols, not of architectures.

**The in-corpus validation split is degenerate.**
`recover_groups` collapses 85,664 of `asl_alphabet`'s 87,000 images into a single near-duplicate component, so a group-disjoint validation split is whatever failed to merge into that blob.
That is why the fusion weight had to be tuned against a held-out corpus instead.
Fixing this split is worth more than most further modelling.

**The geometry model cannot see a hidden thumb.**
It is weakest on M, N, S, V, K and X, which are the handshapes that differ by where a folded thumb sits, exactly where MediaPipe is inferring rather than measuring.
A pixel model can at least read the shading there, which is the argument for keeping it in the blend at all.

**The abstention threshold is corpus-specific.**
The shipped 0.724 was fitted for the pixel model on the holdout corpus, where it gives 90% coverage.
On 37 phone photos from one left-handed signer it covered 54%.
`SLR_THRESHOLD` overrides it, but an override is a judgement call and not a measurement, and this README will not pretend otherwise.

**Nothing here is signer-independent by proof.**
Session recovery is not signer recovery.
The cross-corpus protocol sidesteps that honestly, but it is two corpora, not a population.

## Layout

```
backend/
  src/slr/
    sources.py    corpus registry, downloaders, label normalisation, group-key rules
    leakage.py    perceptual hashing, group recovery, train/test contamination audit
    data.py       group-disjoint and cross-corpus splits, augmentation, torch Dataset
    hands.py      MediaPipe + OpenCV hand detection, cropping, handedness
    landmarks.py  the 21-point geometry classifier, the chirality fold, the offline cache
    fusion.py     probability pooling, ratio selection, protocol row construction
    model.py      18-backbone zoo (5.5M-303M params), layer-wise LR decay, attention rollout
    train.py      fine-tuning loop, and the select-on-validation architecture search
    evaluate.py   temperature scaling, ECE, risk-coverage, per-class confusion
    api.py        research API: pixel, geometry, fused, or both
  tests/          49 tests, no dataset or GPU needed
  notebooks/      experiments.ipynb - the full pipeline end to end on a Colab GPU
app/              the deployable product: one process, API plus a no-build-step web app
dataset/          corpora, manifests, crops, landmark caches (all gitignored)
docs/             26 papers, each with a line saying which decision it supports
```

## Running it

The app is one process serving both the API and the page.

```bash
uv venv --python 3.12 ~/.venvs/slr
VIRTUAL_ENV=~/.venvs/slr uv pip install -r backend/requirements.txt

SLR_WEIGHTS=app/weights/deit3_base_fused_coords.pt \
  PYTHONPATH=backend/src ~/.venvs/slr/bin/python -m uvicorn server.main:app --app-dir app --port 8000
```

Open http://localhost:8000.
The camera panel draws the 21 landmarks live, one colour per finger, at around 25 frames a second.
That overlay is evidence rather than decoration: it tracks the same centre square the classifier is given, through the same detector, so a knuckle that is not marked is a knuckle the model will not be given either.

`GET /api/health` reports the calibration constants and the fusion split.
`POST /api/predict` takes `multipart/form-data` with an `image` field.
`POST /api/landmarks` returns the 21 points for the overlay.

When a prediction falls below the threshold the page still shows the best guess, marked as a guess, with a button to accept it.
Hiding it was worse than useless, because the "could also be" row underneath was still listing the second and third candidates.

Training runs on this MacBook Pro (Apple M4 Pro, 24 GB unified memory, PyTorch MPS).
Open `backend/notebooks/experiments.ipynb` with the `~/.venvs/slr` kernel and work down the cells.

The geometry model does not need any of that.
It trains on a CPU in under a minute, because detection runs once offline into a cache and what is left is a 63-input MLP.

```bash
python -m slr.hands probe dataset/manifests/all.csv --n 400   # measure detection, write nothing
python -m slr.landmarks cache dataset/manifests/all.csv        # 88k images -> a 17 MB npz
python -m slr.landmarks train dataset/landmarks/all.npz
```

Tests:

```bash
PYTHONPATH=backend/src ~/.venvs/slr/bin/python -m pytest backend/tests -q   # 49
PYTHONPATH=backend/src:app ~/.venvs/slr/bin/python -m pytest app/tests -q   # 13
node app/tests/overlay_geometry.mjs                                         # overlay maths
```

The suites need no dataset, no network and no GPU.

## Reading list

`docs/BIBLIOGRAPHY.md` lists 26 papers, each with a line saying which decision in this repository it supports.

The one pair worth quoting first is KAU-SignTransformer (arXiv:2509.03467): **99.02% signer-dependent, 77.71% signer-independent**, same model, same data.
It is recent enough to rebut "better architectures fixed that".

FSboard (arXiv:2407.15806) reports 11.1% CER on signer-disjoint phone fingerspelling from MediaPipe landmarks, which is the prior art for the geometry pipeline here.
