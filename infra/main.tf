# railrank serving infrastructure.
#
# HONEST LABEL, READ THIS FIRST
# -----------------------------
# This module has never been applied. It is validated and planned in CI, and
# it has not been run against a real AWS account, because I am not paying to
# host a portfolio project. Everything below is therefore a design that the
# tooling agrees is well-formed, and not a description of something running.
#
# That statement is deliberate. A README claiming a deployment that does not
# exist is the one thing in this repository that could be taken apart in a
# single question, and "I wrote the module, validated it, and did not apply
# it" is both true and more useful than a vague claim.
#
#   terraform init -backend=false
#   terraform validate
#   terraform plan -var 'image_uri=...' -var 'vpc_id=...' -var 'subnet_ids=[...]'

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ── variables ────────────────────────────────────────────────────────────────

variable "region" {
  type        = string
  default     = "eu-west-2"
  description = "London, because that is where the traffic is."
}

variable "name" {
  type    = string
  default = "railrank"
}

variable "image_uri" {
  type        = string
  description = "The serving image, by digest rather than by tag."
  default     = ""
}

variable "vpc_id" {
  type    = string
  default = ""
}

variable "subnet_ids" {
  type    = list(string)
  default = []
}

variable "desired_count" {
  type    = number
  default = 2
  # Two, not one. A single task means a deploy is an outage and a crash is an
  # outage, and the second task costs less than either.
}

variable "cpu" {
  type    = number
  default = 512
}

variable "memory" {
  type    = number
  default = 1024
}

variable "latency_budget_ms" {
  type        = number
  default     = 25
  description = "The p99 the promotion gate enforces. The alarm below uses the same number, so the budget cannot drift between the gate and production."
}

# ── the image ────────────────────────────────────────────────────────────────

resource "aws_ecr_repository" "this" {
  name                 = var.name
  image_tag_mutability = "IMMUTABLE"
  # Immutable tags, so a deployed version cannot be quietly replaced by a
  # different build under the same name. Rollback then means "deploy the old
  # digest", which works, rather than "rebuild and hope".

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "this" {
  repository = aws_ecr_repository.this.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 20 images; a model image is large."
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = { type = "expire" }
    }]
  })
}

# ── logs ─────────────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "this" {
  name              = "/ecs/${var.name}"
  retention_in_days = 30
}

# ── roles ────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Separate from the execution role and deliberately empty. The execution role
# pulls the image and writes logs; the task role is what the application
# itself can do, and the application reads a model baked into its own image
# and answers questions. It needs nothing.
resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

# ── the service ──────────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "this" {
  name = var.name
}

resource "aws_ecs_task_definition" "this" {
  family                   = var.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = var.name
    image     = var.image_uri
    essential = true
    portMappings = [{ containerPort = 8080, protocol = "tcp" }]
    environment = [
      { name = "RAILRANK_MODEL", value = "/app/artefacts/current" }
    ]
    # The same check the Dockerfile defines: ask the application whether it has
    # a model, not whether the port is open. A process that started, failed to
    # load its model and is answering 503 is not healthy.
    healthCheck = {
      command     = ["CMD-SHELL", "python3 -c \"import sys,urllib.request,json; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)).get('status')=='ok' else 1)\""]
      interval    = 15
      timeout     = 3
      retries     = 3
      startPeriod = 20
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.this.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "ecs"
      }
    }
  }])
}

resource "aws_security_group" "service" {
  name        = "${var.name}-service"
  description = "railrank serving tasks"
  vpc_id      = var.vpc_id

  egress {
    description = "Outbound for image pulls and logs."
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_service" "this" {
  name            = var.name
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  # A deploy adds tasks before it removes any, so the service never drops
  # below its desired count mid-release.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = false
  }

  lifecycle {
    # The image digest is set by the deploy pipeline after the promotion gate
    # passes, not by whoever last ran terraform from a laptop.
    ignore_changes = [task_definition]
  }
}

# ── the alarm that matches the gate ──────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "latency" {
  alarm_name          = "${var.name}-p99-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  threshold           = var.latency_budget_ms / 1000
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  extended_statistic  = "p99"
  period              = 60
  treat_missing_data  = "notBreaching"
  alarm_description   = "p99 above the ${var.latency_budget_ms} ms budget the promotion gate enforces offline. The two numbers are the same variable so they cannot drift apart."
}

# ── outputs ──────────────────────────────────────────────────────────────────

output "ecr_repository_url" {
  value = aws_ecr_repository.this.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.this.name
}
