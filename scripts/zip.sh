#!/bin/bash
zip -r drift-study-2023-01-16.zip . -x "tmp/*" -x "data/*" -x "reports/*" -x ".pytest_cache/*" -x "models/*" -x ".idea/*" -x ".git/*" -x "mlc/*" -x ".vscode/*" -x ".mypy_cache/*" -x ".pytest_cache/*"
