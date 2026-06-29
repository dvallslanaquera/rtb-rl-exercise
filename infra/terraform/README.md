# Terraform (reference stub)

Illustrative infra-as-code for the production topology. **Not** production-ready — review IAM,
VPC/networking, private IP for Cloud SQL/Memorystore, secrets, and deletion protection before
any real `apply`.

Components map to the PoC:

| Component                | Resource                          | Role |
|--------------------------|-----------------------------------|------|
| Feature cache (10ms)     | `google_redis_instance`           | Distributed Redis hot cache |
| Durable feature store    | `google_sql_database_instance`    | Postgres source of truth |
| Serving                  | `google_cloud_run_v2_service`     | Stateless replicas, model hot-swap |
| Images                   | `google_artifact_registry_repository` | Container registry |
| Training + retrain DAG   | Vertex AI (see `../vertex`)       | Custom jobs + Pipelines schedule + Model Registry |

```bash
terraform init
terraform plan  -var project_id=YOUR_PROJECT -var image=REGION-docker.pkg.dev/PROJECT/rtb/rtb-rl:latest
terraform apply -var project_id=YOUR_PROJECT -var image=...
```
