variable "aws_region" {
  description = "AWS region"
  default     = "eu-west-2"
}

variable "environment" {
  description = "Specifies the deployment environment of the resources (e.g., dev, tst, acc, prd)"
  type        = string
  default     = "sandbox"
}

variable "delimiter" {
  description = "Resource name delimiter"
  type        = string
  default     = "-"
}

variable "s3_buckets" {
  description = "A list of S3 Buckets"
  type        = list(any)
  default     = []
}

variable "ecr_repositories" {
  description = "A list of ECR Repositories"
  type        = list(any)
  default     = []
}

variable "apprunner_services" {
  description = "A list of App Runner Services"
  type        = list(any)
  default     = []
}