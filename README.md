# ChurnMaster

Tabular Q-learning for adaptive peer replacement in the routing table of a focal node of a simulated Kademlia-like DHT under churn.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest            # all tests
.\.venv\Scripts\python -m pytest -m "not slow"
```
