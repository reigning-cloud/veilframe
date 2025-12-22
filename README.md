# veilframe

veilframe is a unified Flask web app for LSB encoding (Twitter-compatible) and decoding with a multi-tool analyzer suite. Ships with Docker/Dev Container so all stego tools are preinstalled.

## Features
- Encoder: text or zlib-compressed file payloads embedded into chosen planes (RGB/R/G/B/A) with pre-encoding compression to stay under Twitter's recompression threshold.
- Decoder: binwalk, foremost, strings, exiftool, steghide, zsteg, bit-plane decomposition, optional outguess (deep mode), plus built-in simple LSB and simple zlib extractors. Artifacts are returned as data URLs for inline preview/download.
- UI: encoder/decoder toggle, tooling status panel, bit-plane gallery, downloads, and graceful skip of outguess if it isn't installed.

## Run the app (recommended: Docker Compose)
```bash
cd veilframe_repo
docker compose up --build
```
Then open http://127.0.0.1:5050.

Notes:
- Port 5000 in the container is published to host 5050.
- The repo is bind-mounted into the container for fast iteration.
- Environment: `FLASK_ENV=development`, `FLASK_DEBUG=1`.

## VS Code Dev Container
1. Open the folder in VS Code.
2. Run “Reopen in Container”.
3. The devcontainer uses the same Docker setup; terminals run inside the container with all tools available.

## Local setup/run (no Docker)
Install the stego tools yourself (binwalk, steghide, foremost, outguess, zsteg via Ruby gem, exiftool, strings/binutils, 7z) and Python 3.11+, then:
```bash
cd veilframe_repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
flask --app app run --debug
```
Then open http://127.0.0.1:5000.

## Production run (gunicorn)
```bash
pip install -r requirements.txt
gunicorn wsgi:app --bind 0.0.0.0:8000
```
For platforms like Render, bind to `$PORT` instead:
```bash
gunicorn wsgi:app --bind 0.0.0.0:$PORT
```

## Deploy to Render (Docker)
- Create a new Render Web Service, choose “Docker” as the environment.
- Point it at this repo; Render will build the Dockerfile.
- Render expects the app to listen on `0.0.0.0` and `$PORT` (defaults to 10000 in the CMD). No debug mode in production.
- After deploy, visit the provided Render URL.

## Deploy to Render (render.yaml)
- Push this repo to GitHub and create a new Render Blueprint.
- Render will read `render.yaml` and build using the provided `Dockerfile`, which installs all stego tooling.

## Quick verification (inside container)
```bash
docker compose exec web bash -lc "which binwalk; which foremost; which steghide; which outguess; which zsteg; which exiftool; which strings; which 7z; which file; which unzip; which unsquashfs"
docker compose exec web make smoke
```

If you're running the smoke test in an environment without the stego tools (e.g., restricted CI runners), set `ALLOW_MISSING_TOOLS=1 make smoke` to skip the hard failure while still exercising the encoder.

Smoke test notes: the image installs the app editable (`pip install -e .`) so `import engine` succeeds inside the container.

## Encoder modes
- Simple: one payload (text or file/zlib) into a chosen plane with the classic dropdown.
- Advanced: per-channel payloads (R/G/B/A) encoded in one pass; each channel carries its own text or zlib-compressed file. Enable the advanced toggle, fill per-channel cards, and encode.

## API
- `POST /api/encode` – form-data: `image` (file, required), `payloadMode` (text|zlib), `text` (for text mode), `payload` (file for zlib mode), `plane` (RGB/R/G/B/A). Returns `{ filename, data_url }`.
- `POST /api/decode` – form-data: `image` (file), `password` (optional, used by steghide/outguess), `deep` (true|false to enable outguess if available). Returns `{ results, artifacts }` where artifacts include base64 data URLs for generated bit-plane PNGs and any 7z archives.
- `GET /api/tools` – returns tooling availability used by the UI.

## Troubleshooting
- Port already in use: container binds to host 5050; change the mapped port in `docker-compose.yml` if needed.
- Docker not running: start Docker Desktop/daemon before `docker compose up`.
- Outguess unavailable in your distro: the app will skip it gracefully and mark it missing in the UI; use the Docker image provided here for a working install.
- zsteg depends on the `file` utility; the Docker image installs `file` along with `unzip` and `unsquashfs` (from `squashfs-tools`) so the tooling grid should show them as ✅.
- Slow build: first-time Docker build installs Ruby/gems and stego packages; subsequent runs are faster.

## Security note
Do not expose the Flask debugger PIN or run the dev/debug server on an untrusted network. For production, run behind a proper reverse proxy and disable debug mode (`gunicorn wsgi:app`).

## Twitter compatibility note
The compression safeguard used to keep LSBs intact on Twitter is documented in `docs/twitter-encoding.md`.
