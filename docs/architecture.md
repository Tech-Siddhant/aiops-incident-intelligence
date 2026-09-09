# Architecture: AIOps Incident Intelligence

## Microservice Topology
```
      [ Clients ]
           │
           ▼
    ┌─────────────┐
    │ api_gateway │
    └──────┬──────┘
           ├───► [ auth_service ]
           │
           ▼
    ┌────────────────┐
    │ orders_service │
    └──────┬─────────┘
           │
           ▼
    ┌──────────┐
    │ database │
    └──────────┘
```

## Failure Propagation Path (Database Connection Saturation)
1. **Database Layer (t0)**: Connection pool exhausts (`connection_utilization` -> 1.0, `active_connections` rises), query wait queue builds, database `latency_ms` increases.
2. **Orders Service Layer (t0 + delta_1)**: Blocked on DB connection/query responses; `latency_ms` increases, `error_rate` begins rising due to database query timeouts.
3. **API Gateway Layer (t0 + delta_2)**: Upstream orders requests stall; gateway `latency_ms` rises, HTTP 504/500 error rates spike.
4. **Auth Service**: Operates independently with minimal or normal baseline variation.
