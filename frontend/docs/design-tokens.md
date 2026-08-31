# Design Tokens

The NightmareNet frontend relies on a robust design token system to maintain consistent styling across both light and dark themes. Tokens are implemented natively as CSS custom properties in `src/styles/tokens.css` and are consumed via Tailwind utility classes.

## Color Tokens
Color tokens automatically invert their shades during a theme switch to `.light`.
| Variable | Description | Default Dark Mode | Light Mode |
| -------- | ----------- | ----------------- | ---------- |
| `--color-void` | Deep background color | `#030712` | `#ffffff` |
| `--color-abyss` | Secondary dark tone | `#0a0f1e` | `#f8fafc` |
| `--color-deep` | Deeper card background | `#141b2d` | `#e2e8f0` |
| `--color-surface` | Default surface bg | `#1e293b` | `#cbd5e1` |
| `--color-muted` | Muted background/lines | `#64748b` | `#475569` |
| `--color-text` | Primary text | `#f1f5f9` | `#0f172a` |
| `--color-text-dim` | Dimmed/secondary text | `#94a3b8` | `#334155` |
| `--color-dream` | Primary brand purple | `#818cf8` | - |
| `--color-nightmare` | Destructive brand red | `#f87171` | - |
| `--color-neural` | Cyberpunk cyan accent | `#22d3ee` | - |
| `--color-success` | Success/online green | `#34d399` | - |
| `--color-warning` | Warning yellow | `#fbbf24` | - |

## Spacing Scale
The spacing scale is built on a standard `4px` base.
| Variable | Value | Purpose |
| -------- | ----- | ------- |
| `--space-1` | `4px` | Tiny gaps |
| `--space-2` | `8px` | Standard padding |
| `--space-4` | `16px` | Container padding |
| `--space-8` | `32px` | Section margins |
| `--space-12` | `48px` | Large section spacing |

## Typography
| Variable | Value | Purpose |
| -------- | ----- | ------- |
| `--font-sans` | `Inter`, system | Primary text |
| `--font-mono` | `JetBrains Mono` | Code and data |
| `--text-xs` -> `3xl` | rem-based | Sizing scale |
| `--font-weight-normal` -> `bold` | `400` -> `700` | Weight variants |

## Shadows
Complex multi-layered box shadows with custom RGBA opacities.
| Variable | Purpose |
| -------- | ------- |
| `--shadow-sm` | Small resting shadow |
| `--shadow-md` | Standard elevated element |
| `--shadow-glow` | Large cyan cyberpunk glow |
| `--shadow-panel` | Glassmorphism dashboard panel shadow |
| `--shadow-glow-button` | Primary button interactive glow |
| `--shadow-glow-dream` | Purple card glow |
| `--shadow-glow-nightmare` | Red alert/destructive card glow |
| `--shadow-glow-neural` | Cyan active card glow |

## Motion
Standard easing curves and animation durations to maintain cohesive feedback.
| Variable | Value | Purpose |
| -------- | ----- | ------- |
| `--duration-fast` | `150ms` | Micro-interactions, hover states |
| `--duration-normal` | `250ms` | Standard transitions, modals |
| `--duration-slow` | `400ms` | Large layout shifts, page loads |
| `--ease-standard` | `cubic-bezier(0.4, 0, 0.2, 1)` | Standard smooth easing |
