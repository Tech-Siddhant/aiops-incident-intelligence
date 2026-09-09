# Problem Statement: AIOps Incident Intelligence

## Objective
The AIOps Incident Intelligence platform provides end-to-end incident management for microservice architectures:
1. **Anomaly Detection**: Continuously detect abnormal telemetry patterns across distributed services.
2. **Incident Prediction**: Early warning of impending incidents before user-facing SLA violations occur.
3. **Severity Classification**: Categorize incidents (low, medium, high, critical) based on blast radius and degradation.
4. **Root Cause Analysis (RCA)**: Pinpoint the root cause service and contributing metrics.
5. **Remediation Recommendations**: Provide actionable insights and remediation steps to on-call engineers.

## Scope & Topology
The initial reference topology consists of four interconnected services:
- **API Gateway**: Edge proxy handling incoming client traffic, routing to downstream services.
- **Auth Service**: Authentication and session validation service.
- **Orders Service**: Business logic for order management, calling the Database.
- **Database**: Relational data store supporting order transactions.

## Initial Failure Scenario
- **Database Connection Saturation**: Connection pool exhaustion at the database layer leading to query queuing, elevated database latency, cascading latency to the Orders Service, API Gateway timeout spikes, and downstream HTTP 5xx error rate elevation.
