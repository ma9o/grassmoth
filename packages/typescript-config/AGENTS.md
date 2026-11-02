# TypeScript configuration architecture

- [base.json](base.json) owns shared compiler settings. [nextjs.json](nextjs.json) extends it with Next.js settings. Keep workspace paths, file inclusion and package-specific options in each consumer's `tsconfig.json`.

- Keep this package limited to compiler configuration. Preserve strict typing across consumers when changing shared settings.
