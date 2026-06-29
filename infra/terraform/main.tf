# STUB / reference Terraform for the production topology. Not wired to a live project — review
# IAM, networking and security before any real apply. Mirrors the components named in the PoC:
# Cloud Run (serving), Memorystore Redis (feature cache), Cloud SQL Postgres (durable feature
# store), Artifact Registry (images), and Vertex AI (training / model registry).

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "rtb" {
  location      = var.region
  repository_id = "rtb"
  format        = "DOCKER"
}

# --- Memorystore (Redis) : the distributed feature cache on the 10ms hot path ---
resource "google_redis_instance" "feature_cache" {
  name           = "rtb-feature-cache"
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_gb
  region         = var.region
  redis_version  = "REDIS_7_0"
}

# --- Cloud SQL (Postgres) : durable feature store ---
resource "google_sql_database_instance" "features" {
  name             = "rtb-features"
  database_version = "POSTGRES_16"
  region           = var.region
  settings {
    tier = "db-custom-1-3840"
  }
  deletion_protection = true
}

# --- Cloud Run : stateless serving replicas (load the snapshot from the cache, hot-swap model) ---
resource "google_cloud_run_v2_service" "api" {
  name     = "rtb-bidding-api"
  location = var.region
  template {
    containers {
      image = var.image
      args  = ["rtb", "serve"]
      env {
        name  = "REDIS_URL"
        value = "redis://${google_redis_instance.feature_cache.host}:6379/0"
      }
      env {
        name  = "RTB__STORE__CACHE_BACKEND"
        value = "redis"
      }
      resources {
        limits = { cpu = "2", memory = "2Gi" }
      }
    }
    scaling {
      min_instance_count = 2
      max_instance_count = 50
    }
  }
}

# --- Vertex AI : training jobs + model registry back the every-N-hours retrain loop. ---
# Submit training via google_vertex_ai_* / the aiplatform SDK; schedule with Vertex Pipelines
# Schedules (see infra/vertex/pipeline.py). Left as a documented pointer here.

output "redis_host" {
  value = google_redis_instance.feature_cache.host
}

output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}
