# NeuralScope web (`apps/web`)

Next.js 16 (App Router, Turbopack) + Tailwind v4. Marketing site, research and docs pages, and the `/app` mount point for the research workspace.

```bash
npm install
npm run dev        # http://localhost:3000
npm run lint
npm run typecheck  # next typegen && tsc --noEmit
npm run test       # vitest + Testing Library (jsdom)
npm run build && npm start
```

`/api/*` is proxied to the FastAPI backend at `NEURALSCOPE_API_URL` (default `http://127.0.0.1:8000`), see `next.config.ts`.

## Layout

| Path | What |
| --- | --- |
| `app/globals.css` | Design tokens (`--ns-*`) + Tailwind `@theme` mapping, utilities (`ns-label`, `ns-caps`, `ns-grid-bg`, `ns-hatch`, ...) |
| `app/(marketing)/` | `/`, `/research`, `/docs` with `SiteHeader` / `SiteFooter` |
| `app/app/` | Research workspace (`/app`) and Math Lab (`/app/math`); state in `app/app/layout.tsx` |
| `components/workspace/` | Matrix, inspector tabs (trace, Logit Lens, attention, interventions, diff), Math Lab, charts |
| `lib/api/` | Typed API client, response types, job polling, view cache |
| `components/ui/` | Shared primitives: `EpistemicBadge`, `Panel`, `Button`/`ButtonLink`, `Tabs`/`TabPanel`, `Kbd`, `StatusChip` |
| `components/brand/` | `Logo`/`LogoMark`, `SiteHeader`, `SiteFooter` |
| `components/marketing/` | Landing sections, `HeroTrace` |
| `lib/epistemic.ts` | `EPISTEMIC` map (label, colour var, hex, definition, example) |
| `lib/hero-trace.ts` | Hero snapshot schema + `parseHeroTrace` validation |
| `public/snapshots/hero-trace.json` | Hero trace data: a REAL recorded trace from `scripts/export_hero_snapshot.py` (rebuild after regenerating) |
