# ASL Sign Reader

A web app that reads ASL fingerspelling from a photo, or from a short recording for J, Z and a few word signs.

- `apps/sign-reader` is the FastAPI app and web page.
- `packages/slr` is the library for datasets, training and evaluation.

## Run

```sh
uv sync --frozen --no-dev --all-packages
uv run --no-sync python -m sign_reader.main
```

Then open http://localhost:8000.
The model weights and the hand detector are included.

## Accuracy

Static letters: 91.11% on 720 unique held-out photos.
N is still weak (3/30), and J and Z need the recording mode.
The motion model has only been checked on a small set of clips.
