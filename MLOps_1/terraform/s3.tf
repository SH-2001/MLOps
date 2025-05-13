 resource "aws_s3_bucket" "example" {
  bucket = "test-bucket-sh11"

  tags = {
    Name        = "My bucket"
    Environment = "Dev"
  }
}