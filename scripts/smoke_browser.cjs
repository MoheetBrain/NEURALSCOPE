// Browser smoke test of the V1 acceptance flow (landing → /app → RUN → matrix → Logit Lens → Attention).
// Needs the API (real model) and web running:  node scripts/smoke_browser.cjs   [WEB=… API=… OUT_DIR=…]
// Independent browser smoke test of the acceptance flow against the production build + real API.
const { chromium } = (() => { try { return require('playwright'); } catch { return require(process.env.PLAYWRIGHT_MODULE || '/opt/node22/lib/node_modules/playwright'); } })();
const t0 = Date.now(); const lap = (m) => console.log(`${((Date.now() - t0) / 1000).toFixed(1).padStart(6)}s  ${m}`);
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = []; p.on('pageerror', (e) => errors.push(String(e)));
  await p.goto((process.env.WEB || 'http://localhost:3000') + '/'); lap('landing loaded');
  await p.getByRole('link', { name: /open neural debugger/i }).first().click();
  await p.waitForURL('**/app'); lap('clicked OPEN NEURAL DEBUGGER → /app');
  await p.getByText(/model ready/i).first().waitFor({ timeout: 120000 }); lap('model ready');
  const prompt = p.getByRole('textbox', { name: 'Prompt' });
  await prompt.fill('What is 37 + 58? Give only the answer.');
  const count = async () => (await (await fetch((process.env.API || 'http://127.0.0.1:8000') + '/api/runs?limit=500')).json()).length;
  const before = await count();
  await p.getByRole('button', { name: /^run/i }).first().click(); lap('RUN (runs before: ' + before + ')');
  for (let i = 0; i < 120 && (await count()) === before; i++) await p.waitForTimeout(500);
  const after = await count(); if (after !== before + 1) throw new Error('RUN did not create exactly one new run');
  lap('new run persisted by the API (runs after: ' + after + ')');
  await p.waitForTimeout(1500);
  await p.getByLabel('Generated text').filter({ hasText: '95' }).waitFor({ timeout: 120000 });
  lap('output: ' + (await p.getByLabel('Generated text').innerText()).replace(/\s+/g, ' '));
  const canvas = p.locator('canvas').first(); await canvas.waitFor();
  const box = await canvas.boundingBox();
  await p.mouse.click(box.x + box.width * 0.9, box.y + box.height * 0.2); lap('clicked a matrix cell');
  await p.getByRole('tab', { name: /logit lens/i }).click();
  await p.getByText(/first top-1|top-1 at/i).first().waitFor({ timeout: 60000 }).catch(() => {});
  lap('Logit Lens tab'); await p.screenshot({ path: process.env.OUT_DIR ? process.env.OUT_DIR + '/smoke-lens.png' : 'smoke-lens.png' });
  await p.getByRole('tab', { name: /attention/i }).click(); await p.waitForTimeout(2500);
  lap('Attention tab'); await p.screenshot({ path: process.env.OUT_DIR ? process.env.OUT_DIR + '/smoke-attn.png' : 'smoke-attn.png' });
  console.log(errors.length ? 'PAGE ERRORS: ' + errors.join(' | ') : 'no page errors');
  await b.close();
})().catch((e) => { console.error('SMOKE FAILED', e.message); process.exit(1); });
