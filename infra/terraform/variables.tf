variable "project_id" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type        = string
  default     = "asia-northeast1" # Tokyo
  description = "GCP region"
}

variable "image" {
  type        = string
  description = "Container image for the serving / retrain jobs (Artifact Registry path)"
}

variable "redis_tier" {
  type    = string
  default = "STANDARD_HA"
}

variable "redis_memory_gb" {
  type    = number
  default = 1
}
