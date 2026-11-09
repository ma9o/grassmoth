# grassmoth-server

The study service and execution shell for Grassmoth.

See the [HTTP API](src/grassmoth_server/api/studies.py), [repository documentation](../../docs/README.md),
and [workspace setup](../../docs/guides/dev_setup.md).

For a fresh non-fit workbench study, run `bun run --cwd apps/grassmoth-server study:rebuild`
from the repository root with the local stack running. The
[France rebuild recipe](scripts/dev/france1940simple/README.md) owns its inputs and usage.
