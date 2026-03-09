#!/usr/bin/env bash
flowfigtabminer/bin/python scripts/run_robustness_test.py \
  --input-dir data/input/rebust \
  --tfid-url https://tfid-service-903119038444.us-central1.run.app \
  --figure-url https://figure-service-903119038444.us-central1.run.app \
  --table-url https://table-service-903119038444.us-central1.run.app \
  --output data/output/robustness_report.csv
