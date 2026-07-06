UV := UV_PYTHON=3.12 uv
PANO ?= fixtures/test.jpg
OUT ?= runs/run

.PHONY: setup fetch-weights fixtures test license-guard run determinism-check review clean

setup:
	$(UV) sync
	cd tools/node && npm ci

fetch-weights:
	$(UV) run python tools/fetch_weights.py

fixtures:
	$(UV) run python tools/make_fixtures.py

test:
	$(UV) run pytest

license-guard:
	$(UV) run python tools/license_guard.py

run:
	$(UV) run python -m scenic.run --pano $(PANO) --out $(OUT)

# Acceptance: run the pipeline twice, assert bit-identical manifests.
determinism-check:
	rm -rf runs/_det_a runs/_det_b
	$(UV) run python -m scenic.run --pano $(PANO) --out runs/_det_a
	$(UV) run python -m scenic.run --pano $(PANO) --out runs/_det_b
	$(UV) run python tools/compare_runs.py runs/_det_a runs/_det_b

review:
	open $(OUT)/s8_review/out/index.html

clean:
	rm -rf runs
