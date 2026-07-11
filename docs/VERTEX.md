# Vertex AI Training

Vertex AI Custom Jobs cannot pull directly from GHCR or Docker Hub — the
image must live in **Artifact Registry** in your GCP project. Mirror it
once per release:

```bash
docker build -t wav2vec-mos:v0.1.0 .

gcloud artifacts repositories create wav2vec-mos \
  --repository-format=docker \
  --location=REGION

docker tag wav2vec-mos:v0.1.0 \
  REGION-docker.pkg.dev/PROJECT_ID/wav2vec-mos/wav2vec-mos:v0.1.0

docker push REGION-docker.pkg.dev/PROJECT_ID/wav2vec-mos/wav2vec-mos:v0.1.0
```

Prefer the versioned tag over `latest` so a job always references a known,
reproducible image.

## 0. CI: Push via GitHub Actions

[`.github/workflows/gcp-push.yml`](../.github/workflows/gcp-push.yml) builds
the image and pushes it to Artifact Registry. It's manual
(`workflow_dispatch`) — trigger it from the **Actions** tab and supply the
tag to publish (e.g. `v0.1.0`); it also always pushes a `:<short-sha>` tag.

It authenticates with Workload Identity Federation — no service account key
is stored in GitHub. One-time setup:

```bash
PROJECT_ID=your-project-id
REGION=us-central1
REPO=your-github-org/wav2vec-mos   # exact "owner/repo"

gcloud iam service-accounts create gha-artifact-pusher \
  --project="${PROJECT_ID}"

gcloud artifacts repositories add-iam-policy-binding wav2vec-mos \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --member="serviceAccount:gha-artifact-pusher@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud iam workload-identity-pools create github \
  --project="${PROJECT_ID}" \
  --location=global

gcloud iam workload-identity-pools providers create-oidc github \
  --project="${PROJECT_ID}" \
  --location=global \
  --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="attribute.repository == '${REPO}'"

gcloud iam service-accounts add-iam-policy-binding \
  "gha-artifact-pusher@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')/locations/global/workloadIdentityPools/github/attribute.repository/${REPO}"
```

Then set these as **repository variables** (Settings > Secrets and
variables > Actions > Variables — not secrets, none of these are sensitive):

| Variable | Example |
| --- | --- |
| `GCP_PROJECT_ID` | `your-project-id` |
| `GCP_REGION` | `us-central1` |
| `GCP_ARTIFACT_REPO` | `wav2vec-mos` |
| `GCP_SERVICE_ACCOUNT` | `gha-artifact-pusher@your-project-id.iam.gserviceaccount.com` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github` |

Get the full provider resource name with:

```bash
gcloud iam workload-identity-pools providers describe github \
  --project="${PROJECT_ID}" --location=global \
  --workload-identity-pool=github --format="value(name)"
```

## 1. Choose a Region

Custom Jobs run in one region, and GPU accelerator types are only available
in a subset of regions — and only if your project has quota there. Pick the
region *before* mirroring the image so you push into the matching Artifact
Registry location.

| Accelerator | Example regions |
| --- | --- |
| `NVIDIA_L4` | `us-central1`, `us-west1`, `europe-west4` |
| `NVIDIA_TESLA_A100` | `us-central1`, `us-west1`, `europe-west4` |
| `NVIDIA_H100_80GB` | `us-central1`, `us-east4`, `europe-west4` |

Verify quota before submitting a job — accelerator quota is granted per
region and defaults to 0:

```bash
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --consumer=projects/PROJECT_ID \
  --filter="metric:custom_model_training_nvidia_a100_gpus"
```

Request a quota increase in the console (**IAM & Admin > Quotas**) if it's 0.

## 2. Set the Entrypoint

Custom Jobs are **non-interactive**: they run one command to completion and
the job ends. The image's own `CMD ["bash"]` is a no-op — override
`command`/`args` in the worker pool spec instead.

`worker-pool-spec.yaml`:

```yaml
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: REGION-docker.pkg.dev/PROJECT_ID/wav2vec-mos/wav2vec-mos:v0.1.0
      command: ["bash", "-lc"]
      args:
        - |
          set -e
          cd /workspace/wav2vec-mos
          gsutil -m rsync -r gs://BUCKET/data data
          ( while true; do sleep 300; gsutil -m rsync -r outputs/ gs://BUCKET/outputs/ || true; done ) &
          scripts/train.sh
          gsutil -m rsync -r outputs/ gs://BUCKET/outputs/
      env:
        - name: HF_TOKEN
          value: "hf_..."
        - name: WANDB_API_KEY
          value: "..."
```

Submit:

```bash
gcloud ai custom-jobs create \
  --region=REGION \
  --display-name=wav2vec-mos-train \
  --config=worker-pool-spec.yaml
```

Swap the `args` block to run inference instead: `scripts/infer.sh`.

> Treat `HF_TOKEN` / `WANDB_API_KEY` the same way you'd treat any secret in a
> CLI arg — don't commit `worker-pool-spec.yaml` with real values filled in.
> If your `gcloud`/Vertex AI SDK version supports `secretRef` on container
> env vars, source them from Secret Manager instead of plaintext.

## 3. Handle Storage

Custom Job containers have **ephemeral local disk only** — nothing persists
once the job ends. `data/`, `outputs/`, `.cache/huggingface`, and `wandb/`
all live on that ephemeral disk, so a GCS bucket has to stand in for
persistent storage.

**Approach used above — `gsutil rsync`, no image changes:**

- Pull the dataset in at job start: `gsutil -m rsync -r gs://BUCKET/data data`
- Push `outputs/` out when training finishes.
- Run a background loop that rsyncs `outputs/` every few minutes so
  checkpoints survive a preempted or killed job, not just a clean exit.

This works with the existing image unmodified and is the recommended
default. A `gcsfuse` mount (writing straight to GCS as training runs) would
avoid the rsync loop, but it needs `/dev/fuse` + `SYS_ADMIN`, which Vertex AI
Custom Jobs don't expose to the container — so it isn't a viable option here.

Point `HF_HOME`/`TRANSFORMERS_CACHE` and `WANDB_DIR` at the bucket the same
way if you want the HF cache or wandb run files to survive across jobs;
otherwise each job re-downloads/re-creates them, which is usually fine for
one-off training runs.

## Smoke Test

Submit a throwaway job with the same image and machine spec to confirm the
GPU is visible before running a real training job:

```yaml
containerSpec:
  imageUri: REGION-docker.pkg.dev/PROJECT_ID/wav2vec-mos/wav2vec-mos:v0.1.0
  command: ["python", "-c"]
  args:
    - "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Expected result:

- PyTorch version starts with `2.11.0`
- `torch.cuda.is_available()` prints `True`
- CUDA version is available

## Training

```bash
scripts/train.sh
```

Invoked via the `args` block in the worker pool spec instead of an
interactive shell — see [README.md](README.md) for the local (non-Vertex)
equivalent.
