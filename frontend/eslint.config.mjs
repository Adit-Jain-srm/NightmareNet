import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    rules: {
      "react-hooks/set-state-in-effect": "warn",
      "react-hooks/immutability": "warn",
      "@typescript-eslint/no-require-imports": "warn",
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
  {
    rules: {
      // eslint-config-next/typescript enables this as an "error" by default.
      // A handful of `any`s already exist in the codebase (see
      // docs/development/code-style.md), so it's downgraded to "warn" here
      // to surface every use without blocking unrelated PRs. Once the
      // existing usages are cleaned up, flip this back to "error".
      "@typescript-eslint/no-explicit-any": "warn",
    },
  },
]);

export default eslintConfig;
