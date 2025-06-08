environment = "dev"
aws_region  = "eu-west-2"


s3_buckets = [
  {
    key  = "mlops-course-ehb-data-sh11-2"
    tags = {}
  }
]

s3_buckets = [
  {
    key  = "mlops-course-ehb-mlruns-sh11"
    tags = {}
  }
]

ecr_repositories = [
  {
    key                        = "mlops-course-ehb-repository-sh11"
    image_tag_mutability       = "MUTABLE"
    image_scanning_configuration = {
      scan_on_push = true
    }
    tags = {}
  }
]

apprunner_services = [
  {
    key = "mlops-course-ehb-app-sh11"
    source_configuration = {
      image_repository = {
        image_identifier      = "155677036305.dkr.ecr.eu-west-2.amazonaws.com/ecr-mlops-course-ehb-repository-sh11-dev:latest"
        image_repository_type = "ECR"
        image_configuration = {
          port = 80
        }
      }
      auto_deployments_enabled = true
    }
    tags = {}
  }
]