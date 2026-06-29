"""Vertex AI Pipelines DAG (STUB).

Maps the local retrain DAG (:mod:`rtb_rl.pipelines.retrain_loop`) onto GCP-native components:
build-features -> train (Vertex custom training job, GPU optional) -> sim-gate -> conditional
promote into the Vertex AI Model Registry, fronted by Cloud Run for serving with Memorystore
(Redis) as the feature cache.

This file is intentionally not wired to a live project; it documents the production topology.
Compile with KFP and submit via ``google-cloud-aiplatform``::

    from kfp import compiler
    compiler.Compiler().compile(retrain_pipeline, "retrain_pipeline.json")
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT, location=REGION)
    aiplatform.PipelineJob(display_name="rtb-retrain",
                           template_path="retrain_pipeline.json").submit()

Schedule every N hours with a Vertex Pipelines Schedule (or Cloud Scheduler -> Pub/Sub ->
Cloud Functions trigger).
"""

from __future__ import annotations

# NOTE: kfp is not a project dependency; this is a reference skeleton.
try:
    from kfp import dsl
except Exception:  # noqa: BLE001 - stub importable without kfp installed
    dsl = None


PROJECT = "your-gcp-project"
REGION = "asia-northeast1"  # Tokyo
IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT}/rtb/rtb-rl:latest"


def _component(func):
    return dsl.container_component(func) if dsl is not None else func


if dsl is not None:

    @dsl.container_component
    def build_features_op():
        return dsl.ContainerSpec(image=IMAGE, command=["rtb"], args=["build-features"])

    @dsl.container_component
    def train_op():
        return dsl.ContainerSpec(image=IMAGE, command=["rtb"], args=["train"])

    @dsl.container_component
    def retrain_once_op():
        # build-features -> warm-start train -> sim-gate -> conditional promote
        return dsl.ContainerSpec(image=IMAGE, command=["rtb"], args=["retrain", "--once"])

    @dsl.pipeline(name="rtb-retrain", description="Every-N-hours RTB DQN retrain + sim-gated promote")
    def retrain_pipeline():
        retrain_once_op()


def describe() -> dict:
    """Return a JSON-serializable description of the production topology (for docs/tests)."""
    return {
        "project": PROJECT,
        "region": REGION,
        "image": IMAGE,
        "stages": ["build-features", "warm-start-train", "sim-gate", "conditional-promote"],
        "serving": "Cloud Run + Memorystore(Redis) + Vertex AI Model Registry",
        "schedule": "Vertex Pipelines Schedule every retrain.interval_hours",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), indent=2))
