

# ml-engineering

Kubernetes-native machine learning and data engineering project demonstrating **ETL pipelines, workflow orchestration, CI/CD, and MLOps practices** using Argo Workflows, Docker, and GitHub Actions.

This repository showcases a **production-style data pipeline** used to ingest, process, and prepare large-scale crypto market data, designed with cloud-native and DevOps best practices.

---

##  Project Overview

The project implements a **daily automated ETL workflow** that:

* Fetches raw market data 
* Cleans, validates, and aggregates large CSV datasets
* Produces processed datasets ready for analytics or ML training
* Runs on a **scheduled, containerized workflow** using **Argo CronWorkflows**


---

## 🏗️ Architecture

```
┌──────────────┐
│ GitHub       │
│  (Source)    │
└──────┬───────┘
       │
       │ GitHub Actions (CI)
       ▼
┌──────────────────────┐
│ Docker Images        │
│  fetcher / processor│
└──────┬───────────────┘
       │
       │ Argo CD / kubectl
       ▼
┌─────────────────────────────┐
│ Argo CronWorkflow (Daily)   │
│                             │
│  ┌──────────┐  ┌─────────┐ │
│  │ Fetch    │→ │ Process │ │
│  │ Data     │  │ ETL     │ │
│  └──────────┘  └─────────┘ │
│                             │
└─────────────────────────────┘
       │
       ▼
Processed datasets (CSV / Parquet-ready)
```

---



---

## 🔄 ETL Pipeline Details

### 1️⃣ Fetch Stage

* Downloads raw Binance futures kline data
* Stores data in a structured raw-data directory
* Implemented as a **containerized Python job**

### 2️⃣ Process Stage

* Validates CSV files
* Handles missing / malformed data
* Aggregates large datasets efficiently using pandas
* Outputs cleaned, processed datasets



---

## ⏱️ Workflow Orchestration (Argo)

* Uses **Argo Workflows & CronWorkflows**
* Runs **daily scheduled ETL jobs**
* todo:

  * Retry strategies
  * Step dependencies
  * Failure visibility via Argo UI



---

## 🚀 CI/CD & Automation

* **GitHub Actions** used for:

  * Docker image builds
  * Dependency installation
  * Image versioning
* Images are designed to be:

  * Reproducible
  * Stateless
  * Environment-agnostic

---

## 📊 Observability & Reliability

* todo:Designed to integrate with:

  * Prometheus metrics (workflow status, duration)
  * Grafana dashboards
* Clear failure modes:

  * Missing input data
  * Permission / volume mount issues
  * Data validation errors

---

## 🔐 Security & Best Practices

* Containerized workloads (no mutable hosts)
* Principle of least privilege for data directories
* Clear separation of:

  * Raw data
  * Processed data
  * Pipeline logic
* GitOps-friendly workflow definitions

---

## Local Development

 

It is intentionally designed as a **portfolio-grade project** for:

* DevOps Engineer
* Platform Engineer
* Cloud Engineer
* MLOps / Data Platform roles

---

## 📌 Future Improvements

* Add Parquet output support
* Integrate S3 / GCS storage backends
* Add data quality checks (Great Expectations)
* Extend MLflow training pipelines
* Add SLA / SLO definitions for workflows

