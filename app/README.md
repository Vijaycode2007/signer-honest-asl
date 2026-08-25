# app/ - Sign Reader

A hostable web app that reads ASL fingerspelling from a photo.
Point a camera at a hand shape, and it appends a letter to a message you build up one photo at a time.
When it is not confident enough, it refuses to guess and says so.

This is the deployable product.
`backend/` remains the research code, and this app imports it rather than copying it, so what is served is what was measured.
`backend-v2/` is untouched by this app.

## The model is not a choice this app makes

`architecture_sweep_results.pdf` settled it.
DeiT III base (`deit3_base_patch16_224.fb_in22k_ft_in1k`, 85.84M parameters, 224px) won a validation-only run-off against seven other backbones at 0.9456 macro-F1.
The preset already existed in `backend/src/slr/model.py`; this app pins to it and never trains at request time.

Architecture selection and deployment evaluation are separate.
DeiT III's 0.9456 is its architecture-selection **validation** score; the UI's protocol figures now come from the selected 0.62 pixel / 0.38 geometry strategy.
Its cross-corpus headline is the untouched 1,218-photo final score, not the 522-photo partition used to select the ratio.

## Layout

```
app/
  server/
    engine.py       load the bundle, answer one photo, attention rollout
    captures.py     keep the photos you take, filed under the label you type
    main.py         the JSON endpoints plus the static site
  web/              the frontend: no framework, no build step
  design/           the design canvas the UI was drawn from
  tests/            checks for the serving path
  train.py          produce the weights, once
  model_card.json   the measured figures the UI quotes
  Dockerfile        build from the REPO ROOT, not from app/
  render.yaml       Render blueprint
  weights/          the bundle lands here, gitignored
```

## Running it locally

Weights are not in git.
Get a bundle first (see below), then:

```bash
PYTHONPATH=backend/src python -m uvicorn server.main:app --app-dir app --port 8000
```

Open http://localhost:8000.
The camera needs a secure context, so `localhost` works but a bare LAN IP will not: browsers refuse `getUserMedia` over plain HTTP on a non-localhost origin.
The "Upload instead" button works everywhere and takes the same path through the model.

Tests:

```bash
PYTHONPATH=backend/src:app pytest app/tests -q
```

## Producing the weights, once

This is the step that makes every later start free.
Training runs once, writes one file, and the server only ever loads it.

```bash
python app/train.py dataset/manifests/all.csv --epochs 8
```

That writes `app/weights/deit3_base.pt`: weights, class list, the fitted temperature, and the abstention threshold, all in one bundle.
Keeping calibration inside the checkpoint means the threshold the service abstains on is provably the one whose error rate was measured.

Two things worth knowing:

- Run it on a GPU.
  A base-scale ViT is roughly 25 minutes per epoch on a T4 and effectively never finishes on a laptop CPU.
  This Mac is an inference box; Colab is the training box.
- The operating point is read off the **validation** split by default (`--threshold-split val`).
  Picking it by looking at test scores would reintroduce, at the level of the deployment, the leakage this project exists to expose.

Already have a finished run and just want the bundle?

```bash
python app/train.py --from-run runs/deploy_deit3_base dataset/manifests/all.csv
```

## Deploying to Render

`render.yaml` is a blueprint: point Render at the repo and it reads the file.

Host the bundle anywhere that serves a plain GET (a GitHub release asset, a Hugging Face Hub file, a signed bucket URL) and set `SLR_WEIGHTS_URL`.
The server downloads it on first boot and caches it on the mounted disk, so restarts do not re-download it.
Weights are deliberately not baked into the image: 172 MB of checkpoint does not belong in a Docker layer or in git.

Building by hand uses the repo root as context, because the image needs both this app and the `slr` package above it:

```bash
docker build -f app/Dockerfile -t sign-reader .
docker run -p 8000:8000 -e SLR_WEIGHTS_URL=https://... sign-reader
```

### Memory, honestly

DeiT III base is 85.84M parameters.
The bundle ships fp16 at about 172 MB, but CPU inference wants fp32, so the loaded weights are about 343 MB and the torch runtime around them wants roughly another 250 MB.

**It will not fit Render's 512 MB tiers.**
The free and starter plans OOM on boot.
`standard` (2 GB) is the smallest plan that runs this model, which is why the blueprint asks for it.

If you need it on a smaller box, the honest options are to serve a smaller backbone from the same sweep (EfficientNetV2-S scored 0.8426 at 20.2M parameters) or to export to ONNX Runtime with int8 dynamic quantization.
Neither is built here, and the second would cost you the attention heatmap, which needs the PyTorch attention matrices.

## Configuration

| variable | meaning |
| --- | --- |
| `SLR_WEIGHTS` | local bundle path. If unset, prefers `deit3_base_fused.pt` when present, otherwise `deit3_base.pt` |
| `SLR_WEIGHTS_URL` | fetched on first boot when the local file is missing |
| `SLR_THRESHOLD` | override the abstention threshold. Prefer the measured one |
| `SLR_CAPTURES` | where labelled photos are collected. Default `dataset/raw/mine`, and collection is off when that directory's parent does not exist |
| `SLR_SESSION` | the sitting the next photos belong to. Default is today's date |
| `PORT` | listen port. Render sets this |

## API

| route | what it does |
| --- | --- |
| `GET /api/health` | readiness, device, class count, calibration constants |
| `GET /api/model` | the model card the "How it works" panel renders |
| `POST /api/predict` | multipart `image`, plus `top_k` and `explain` query params |
| `POST /api/landmarks` | multipart `image` -> the 21 hand points, for the live overlay |
| `POST /api/label` | `{capture_id, label}` - file a photo you just took under the sign it really was |
| `GET /api/captures` | how much has been collected, and under which labels |

`POST /api/predict?explain=true` also returns `attention_png_b64`, a translucent PNG of the attention rollout that the page overlays on the photo.
It is there to catch the failure that accuracy alone hides: a model attending to the wall behind the signer rather than the hand.

## Fusing pixels with hand geometry

The fused bundle adds a 35-feature per-finger geometry MLP to the pixel model.
Both predict the same 29 classes; their calibrated probabilities are combined by a linear pool.

The sweep tested 101 pixel weights from 0.00 through 1.00.
Selection maximised cross-corpus tuning macro-F1 while allowing at most a 0.01 macro-F1 loss against pixels on Protocols A and B.
The selected ratio is **0.62 pixel / 0.38 geometry**.
Protocol C was split by capture group into 522 tuning and 1,218 untouched final photos; the partitions share no paths or groups and both contain all 29 classes.

| untouched test split | pixels macro-F1 | fused macro-F1 | change |
| --- | ---: | ---: | ---: |
| A: random | 1.0000 | 1.0000 | +0.0000 |
| B: group | 0.8109 | 0.8135 | +0.0027 |
| C: cross-corpus | 0.6572 | 0.7970 | +0.1398 |

The frozen twelve-photo `dataset/manifests/mine.csv` set was test-only and never used to choose the ratio.
Pixels got 9/12; geometry and the fixed fused ratio each got 7/12.
The per-photo predictions and all 101 sweep rows are in `runs/fusion/fusion_report.json`.
Serve the selected bundle with `SLR_WEIGHTS=app/weights/deit3_base_fused.pt`.

## The live skeleton

The camera panel draws MediaPipe's 21 landmarks on the preview as you move, one colour per finger, at around 25 frames a second on localhost.

It is evidence, not decoration.
The frame it tracks is the same centre square `/api/predict` is given, and the detector is the same one `slr.hands` uses, so a knuckle that is not marked is a knuckle the model will not be given either.
A chip in the corner says `Tracking hand` or `Looking for a hand`, which turns "why is it not reading my sign" from a guess into something you can see.

Detection runs on the server, not in the browser.
That costs a round trip per frame and ties the frame rate to the box, and it buys exact agreement with what the prediction path sees, no second copy of the model to keep in step, and no build step.
Over a real network it will lag; the upgrade then is `@mediapipe/tasks-vision` in WASM on the client, which is a vendored asset rather than a rewrite.

The overlay is hidden while a captured still is on screen, because those bones described the live frame and not the frozen one.

`toFrame` in `app/web/app.js` composes the centre crop with `object-fit: cover`, and getting either backwards puts the skeleton beside the hand rather than on it, which reads as a broken detector. There is a dependency-free check for it:

```bash
node app/tests/overlay_geometry.mjs
```

## Collecting your own photos to fine-tune on

Every photo the app reads is kept, and the box under the answer files it under the sign it really was:

```
dataset/raw/mine/<session>/<LABEL>/<id>.jpg
```

That is already the layout `slr.sources` walks, so the collection is one command away from a training manifest:

```bash
PYTHONPATH=backend/src python -m slr.sources manifest mine --out dataset/manifests/mine.csv
```

The session directory is the group key, and it is the reason this is worth doing carefully.
One sitting shares a light, a shirt and a camera distance, so its frames are near-copies of each other.
`slr.data` splits on the group, which keeps a whole sitting on one side of the boundary - the thing that stops a fine-tune on this data from reproducing exactly the leakage Protocol A exists to expose.
So collect across several days, change something each time, and set `SLR_SESSION` when you start a new one.

Two things this data is not.
It is one signer's hand, so anything measured on it is signer-dependent by construction and belongs in a write-up as adaptation, never as a headline accuracy.
And the labels are only as correct as the handshapes, so a letter you sign wrong is a letter the model learns wrong.

Photos you have not labelled yet wait in `_inbox/`.
They are skipped by the manifest builder, which says how many are waiting, and they can be labelled later by dragging them into the right class folder by hand.

Collection is off unless `SLR_CAPTURES` is set or `dataset/raw` exists beside the source tree, so a deployed copy never banks its visitors' photographs.

A missing checkpoint does not crash the process.
The site still comes up, `/api/health` reports why it is not ready, and the page says so in plain words, because a crash-looping container tells whoever deployed it nothing.
