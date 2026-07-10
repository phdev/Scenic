UV := UV_PYTHON=3.12 uv
PANO ?= fixtures/test.jpg
OUT ?= runs/run
FIXTURE ?= fixtures/ci_tiny.jpg

.PHONY: setup fetch-weights fixtures test license-guard run determinism-check review accept clean sweep

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

# Promote a completed run to the runs/_accepted baseline that s8_review
# compares against. Refuses shippable=false runs unless FORCE=1 (FORCE=0 is
# off, not on — filter-out guards against truthy-empty confusion).
accept:
	@test -n "$(RUN)" || { echo "usage: make accept RUN=runs/<name> [FORCE=1]"; exit 2; }
	$(UV) run python tools/accept_run.py $(RUN) $(if $(filter-out 0,$(FORCE)),--allow-failed-gates,)

clean:
	rm -rf runs

# Deterministic parameter sweep over {s4.scale_multiplier, s4.base_stride,
# s3.edge_depth_ratio_min, s3.band_px_max}; ranked report -> runs/_sweep/.
sweep:
	$(UV) run python tools/sweep.py --pano $(FIXTURE)
