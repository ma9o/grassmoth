import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
import functional from "eslint-plugin-functional";
import sonarjs from "eslint-plugin-sonarjs";
import unicorn from "eslint-plugin-unicorn";
import tseslint from "typescript-eslint";

// Library values the client receives but does not own. Their declarations are mutable, so the
// immutability rules treat them as immutable values instead of asking for readonly copies.
const libraryTypes = [
  {
    from: "lib",
    name: [
      "Element",
      "HTMLElement",
      "HTMLCanvasElement",
      "HTMLDivElement",
      "SVGElement",
      "SVGSVGElement",
      "SVGGElement",
      "SVGRectElement",
      "CanvasRenderingContext2D",
      "Event",
      "PointerEvent",
      "KeyboardEvent",
      "MouseEvent",
      "Uint8Array",
      "Request",
      "Response",
      "URL",
      "AbortSignal",
      "DOMRect",
      "Promise",
    ],
  },
  {
    from: "package",
    package: "react",
    name: [
      "ReactNode",
      "ReactElement",
      "ReactPortal",
      "RefObject",
      "Ref",
      "PointerEvent",
      "KeyboardEvent",
      "MouseEvent",
      "FocusEvent",
      "ChangeEvent",
      "WheelEvent",
      "SyntheticEvent",
      "CSSProperties",
      "ComponentProps",
      "ComponentPropsWithoutRef",
      "HTMLAttributes",
      "SVGProps",
    ],
  },
  {
    // @tanstack/react-table re-exports these from table-core, where they are declared.
    from: "package",
    package: "@tanstack/table-core",
    name: [
      "Table",
      "Header",
      "HeaderGroup",
      "Column",
      "Row",
      "Cell",
      "ColumnDefBase",
      "GroupColumnDefBase",
      "AccessorFnColumnDefBase",
      "AccessorKeyColumnDefBase",
      "SortingState",
      "Updater",
    ],
  },
  {
    from: "package",
    package: "@tanstack/react-query",
    name: ["QueryClient", "UseQueryResult", "QueryKey"],
  },
  { from: "package", package: "ai", name: ["UIMessage"] },
  { from: "package", package: "elkjs", name: ["ElkNode"] },
];
// Aliases of instantiated library generics, which the package form cannot match, go by name.
const libraryAliases = ["LucideIcon", "ClassArray", "ClassDictionary"];

const eslintConfig = defineConfig([
  js.configs.recommended,
  ...nextVitals,
  ...nextTs,
  {
    files: ["**/*.{ts,tsx,mts,cts}"],
    extends: [
      tseslint.configs.strictTypeChecked,
      tseslint.configs.stylisticTypeChecked,
      unicorn.configs.recommended,
    ],
    plugins: { functional, sonarjs },
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    settings: {
      immutability: {
        overrides: [
          ...libraryTypes.flatMap(({ name, ...source }) =>
            name.map((typeName) => ({ type: { ...source, name: typeName }, to: "ReadonlyDeep" })),
          ),
          ...libraryAliases.map((name) => ({ type: name, to: "ReadonlyDeep" })),
          // Inside a union, the plugin sees TanStack's row as an import() path, not by name.
          { type: /@tanstack\/table-core[^"]*"\)\.Row<.+>$/u, to: "ReadonlyDeep" },
        ],
      },
    },
    rules: {
      "@typescript-eslint/no-unsafe-type-assertion": "error",
      "@typescript-eslint/consistent-type-assertions": ["error", { assertionStyle: "never" }],
      // Lint owns exhaustiveness, so an exhaustive switch carries no default branch.
      "@typescript-eslint/switch-exhaustiveness-check": [
        "error",
        { allowDefaultCaseForExhaustiveSwitch: false, requireDefaultForNonUnion: true },
      ],
      // Functions never mutate their inputs, and the client's types are immutable values.
      "@typescript-eslint/prefer-readonly-parameter-types": [
        "error",
        {
          ignoreInferredTypes: true,
          treatMethodsAsReadonly: true,
          allow: [...libraryTypes, ...libraryAliases],
        },
      ],
      "functional/type-declaration-immutability": [
        "error",
        { rules: [{ identifiers: "^.+$", immutability: "ReadonlyDeep", comparator: "AtLeast" }] },
      ],
      "react/prefer-read-only-props": "error",
      // Canvas drawing configures the context and sizes the element it is given.
      "no-param-reassign": [
        "error",
        { props: true, ignorePropertyModificationsFor: ["context", "canvas"] },
      ],
      // The same cognitive complexity measure and limit that complexipy reports for Python.
      "sonarjs/cognitive-complexity": ["error", 15],
      "max-lines": ["error", { max: 400, skipBlankLines: true, skipComments: true }],
      "@typescript-eslint/max-params": ["error", { max: 4 }],
      // The API encodes absent values as null.
      "unicorn/no-null": "off",
      "unicorn/prevent-abbreviations": "off",
      // Next.js route folders are named after camelCase params, such as [workspaceId].
      "unicorn/filename-case": "off",
      // No-op callbacks are written `() => undefined`, since no-empty-function forbids `() => {}`.
      "unicorn/no-useless-undefined": "off",
      // The tsconfig targets ES2017, which has no bigint literals.
      "unicorn/prefer-bigint-literals": "off",
      // Not type-aware: it rewrites Array.from on ArrayLike values, which cannot be spread.
      "unicorn/prefer-spread": "off",
      // Biome owns literal formatting and writes hex digits in lowercase.
      "unicorn/number-literal-case": "off",
      // Numbers interpolate as ids and coordinates; `never` only reaches exhaustiveness errors.
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        { allowNumber: true, allowNever: true },
      ],
      // Arrow-shorthand event handlers may return the void call they wrap.
      "@typescript-eslint/no-confusing-void-expression": ["error", { ignoreArrowShorthand: true }],
      // An empty string or absent flag reads as absent; a zero count must be compared explicitly.
      "@typescript-eslint/strict-boolean-expressions": [
        "error",
        { allowNullableString: true, allowNullableBoolean: true, allowNumber: false },
      ],
      "@typescript-eslint/method-signature-style": ["error", "property"],
      "@typescript-eslint/no-shadow": "error",
      "@typescript-eslint/no-import-type-side-effects": "error",
      // One-line guards may omit braces; a body the formatter wraps needs them.
      curly: ["error", "multi-line"],
      eqeqeq: ["error", "always", { null: "ignore" }],
      "no-console": "error",
      "no-nested-ternary": "error",
      "object-shorthand": "error",
      // A named function expression gives memo and forwardRef components their display name.
      "prefer-arrow-callback": ["error", { allowNamedFunctions: true }],
      "import/first": "error",
      "import/newline-after-import": "error",
      // The SCC pre-pass parses every imported library despite ignoreExternal, about ten times
      // the cost of the cycle search; the search alone skips libraries.
      "import/no-cycle": ["error", { ignoreExternal: true, disableScc: true }],
      "import/no-duplicates": "error",
      "react/button-has-type": "error",
      "react/jsx-no-constructed-context-values": "error",
      "react/no-array-index-key": "error",
      "react/no-unstable-nested-components": "error",
    },
  },
  {
    // Vendored shadcn primitives forward the UI library's own prop types, which are not readonly.
    files: ["src/components/ui/{accordion,badge,button,skeleton,table,tooltip}.tsx"],
    rules: { "@typescript-eslint/prefer-readonly-parameter-types": "off" },
  },
  {
    files: ["src/lib/{dag,model-asset,admission,tables}/**/*.{ts,tsx}"],
    ignores: ["**/*.test.ts"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/components/**", "**/components/**"],
              message: "Shared UI logic must not depend on rendering components.",
            },
          ],
        },
      ],
    },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
