PY ?= python
CKPT ?= weights/forknet_reds.pth

.PHONY: install test test-legacy complexity train train-legacy test-reds4 test-lp ablations tables

install:            ## install package and dev dependencies
	$(PY) -m pip install -e ".[dev,video]"

test:               ## unit, integration and end-to-end tests (CPU, ~3 min)
	$(PY) -m pytest

test-legacy:        ## equivalence with the original code (needs PIXEL_LEVEL_VSR_ROOT)
	$(PY) -m pytest tests/test_legacy_equivalence.py -rs

complexity:         ## params / MACs / FLOPs of the model and ablations
	$(PY) scripts/complexity.py --out results/complexity.json

train:              ## main model, paper hyper-parameters, corrected propagation
	$(PY) scripts/train.py --config configs/train_reds.yaml

train-legacy:       ## exact behaviour of the original training code
	$(PY) scripts/train.py --config configs/train_reds_legacy.yaml

ablations:          ## Table 1
	for c in first_order no_convnext no_bsconv legacy_propagation; do \
		$(PY) scripts/train.py --config configs/ablation/$$c.yaml; done

test-reds4:         ## REDS4, standard and original protocol
	$(PY) scripts/test.py --config configs/test_reds4.yaml --checkpoint $(CKPT) --protocol both

test-lp:
	$(PY) scripts/test.py --config configs/test_lp.yaml --checkpoint $(CKPT) --protocol both

tables:
	$(PY) scripts/make_tables.py results/*/*/metrics.json --out results/tables.md
