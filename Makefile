quality:
	black --required-version 23 --check .
	ruff .

style:
	black --required-version 23 .
	ruff . --fix