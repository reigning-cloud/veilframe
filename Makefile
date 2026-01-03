.PHONY: smoke

smoke:
	@PYTHONPATH=. python -c "import engine, sys; print(getattr(engine, '__file__', None))"
	@PYTHONPATH=. python scripts/smoke.py
