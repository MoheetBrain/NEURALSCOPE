// SelfLab browser acceptance (SA6 independent validation).
//
// Flow: /app/selflab → chat (Load example → SEND TO TARGET) → select a claim span → INVESTIGATE CLAIM →
//       SUGGEST TESTS → COMPARE CONTEXTS (commit + run the case test) → battery template (commit + run, budget-capped)
//       → evidence → inspect a trial → OPEN IN LAB (trial replay) → open a Lab run (/app?run=) → EXPORT preview.
//
// Two modes (MODE=fixture|real; default: detected from GET /api/selflab/status):
//   fixture — DEMO random-weight model. Asserts the DEMO/FIXTURE marking is visible on the page, the case and the
//             evidence; the target's text is meaningless, so the span is chosen by offsets.
//   real    — a real model served by someone else (the coordinator). Asserts NO fixture marking. Never starts servers.
//
// Usage (servers must already be running):
//   WEB=http://localhost:3106 API=http://127.0.0.1:8106 OUT_DIR=/tmp/selflab-smoke node scripts/smoke_selflab_browser.cjs
// Optional: MODE, SMOKE_MAX_TRIALS (battery budget, default 6), STEP_TIMEOUT_MS (default 120000),
//           PLAYWRIGHT_MODULE (path to a playwright package whose bundled Chromium is cached locally).
//
// Every step is recorded (pass / fail / known_defect) with timing and a screenshot; a JSON summary is written to
// OUT_DIR/summary.json. Exit code 1 if any REQUIRED step failed. Steps that hit a defect already recorded in
// docs/selflab/VALIDATION_REPORT.md are reported as "known_defect" (and still fail the run if STRICT=1).
'use strict';
const fs = require('fs');
const path = require('path');

const { chromium } = (() => {
  const tries = [process.env.PLAYWRIGHT_MODULE, 'playwright', '/opt/node22/lib/node_modules/playwright'].filter(Boolean);
  for (const t of tries) { try { return require(t); } catch { /* next */ } }
  throw new Error('playwright not found: set PLAYWRIGHT_MODULE to a playwright package directory');
})();

const WEB = (process.env.WEB || 'http://localhost:3106').replace(/\/+$/, '');
const API = (process.env.API || 'http://127.0.0.1:8106').replace(/\/+$/, '');
const OUT_DIR = path.resolve(process.env.OUT_DIR || 'smoke-selflab-out');
const MAX_TRIALS = String(process.env.SMOKE_MAX_TRIALS || '6');
const STEP_TIMEOUT = Number(process.env.STEP_TIMEOUT_MS || 120000);
const STRICT = process.env.STRICT === '1';
fs.mkdirSync(OUT_DIR, { recursive: true });
for (const f of fs.readdirSync(OUT_DIR)) if (/^\d\d-.*\.png$|^summary\.json$/.test(f)) fs.unlinkSync(path.join(OUT_DIR, f));

const t0 = Date.now();
const results = [];
const lap = (m) => console.log(`${((Date.now() - t0) / 1000).toFixed(1).padStart(7)}s  ${m}`);
const apiGet = async (p) => { const r = await fetch(API + p); if (!r.ok) throw new Error(`GET ${p} → ${r.status}`); return r.json(); };

(async () => {
  const status = await apiGet('/api/selflab/status');
  const MODE = process.env.MODE || (status.is_fixture ? 'fixture' : 'real');
  lap(`mode=${MODE} (backend is_fixture=${status.is_fixture}, model=${status.model && status.model.provenance && status.model.provenance.model_id})`);

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  page.setDefaultTimeout(STEP_TIMEOUT);
  const pageErrors = [];
  page.on('pageerror', (e) => pageErrors.push(String(e)));
  const calls = [];
  page.on('response', async (r) => {
    const u = r.url();
    if (u.includes('/api/selflab/') && r.request().method() !== 'GET') {
      calls.push({ method: r.request().method(), path: u.replace(/^.*?\/api\//, '/api/'), status: r.status(),
                   body: (await r.text().catch(() => '')).slice(0, 400) });
    }
  });
  const lastCall = (re) => [...calls].reverse().find((c) => re.test(c.path));
  let shot = 0;
  const snap = async (name) => {
    const f = path.join(OUT_DIR, `${String(++shot).padStart(2, '0')}-${name}.png`);
    await page.screenshot({ path: f, fullPage: true }).catch(() => {});
    return f;
  };
  const bodyText = () => page.locator('body').innerText();

  async function step(name, fn, { required = true, knownDefect = null } = {}) {
    const s = Date.now();
    const rec = { name, required, status: 'pass', seconds: 0, detail: null, screenshot: null, known_defect: null };
    try {
      rec.detail = (await fn()) ?? null;
    } catch (e) {
      rec.status = knownDefect && knownDefect.test(String(e && e.message)) ? 'known_defect' : 'fail';
      rec.detail = String(e && e.message ? e.message : e).slice(0, 600);
      if (rec.status === 'known_defect') rec.known_defect = knownDefect.source;
    }
    rec.seconds = +((Date.now() - s) / 1000).toFixed(2);
    rec.screenshot = await snap(name);
    results.push(rec);
    lap(`${rec.status.toUpperCase().padEnd(12)} ${name}${rec.detail ? ' — ' + JSON.stringify(rec.detail).slice(0, 220) : ''}`);
    return rec.status === 'pass';
  }

  // ------------------------------------------------------------------ 1. page + marking
  await step('open-selflab', async () => {
    await page.goto(WEB + '/app/selflab', { waitUntil: 'networkidle' });
    await page.getByText(/research prototype/i).first().waitFor();
    const txt = await bodyText();
    const demoBanner = /DEMO\s*[—-]\s*RANDOM-WEIGHT FIXTURE MODEL/i.test(txt);
    const fixtureChip = /DEMO \/ FIXTURE/i.test(txt);
    if (MODE === 'fixture' && !(demoBanner && fixtureChip)) throw new Error(`fixture marking missing (banner=${demoBanner}, chip=${fixtureChip})`);
    if (MODE === 'real' && (demoBanner || fixtureChip)) throw new Error('real mode but the page shows DEMO/FIXTURE marking');
    return { demoBanner, fixtureChip };
  });

  // ------------------------------------------------------------------ 2. chat
  const chatOk = await step('chat-send', async () => {
    await page.getByRole('button', { name: /load example/i }).click();
    const q = await page.getByLabel('Question').inputValue();
    if (!q.trim()) throw new Error('example did not fill the question');
    await page.getByRole('button', { name: /send to target/i }).click();
    await page.getByText(/stopped by/i).first().waitFor();
    const c = lastCall(/selflab\/chat$/);
    if (!c || c.status !== 200) throw new Error(`POST /chat → ${c && c.status}`);
    const runLink = page.getByRole('button', { name: /^run [0-9a-f]{6}/i }).first();
    await runLink.waitFor();
    return { run: (await runLink.innerText()).trim() };
  });

  // ------------------------------------------------------------------ 3. select span + investigate
  const caseOk = chatOk && await step('select-span-and-investigate', async () => {
    const whole = page.getByRole('button', { name: /whole response/i });
    let how;
    if (await whole.isEnabled()) {
      await whole.click();
      how = 'whole response';
    } else { // DEMO model emits whitespace: choose the span by offsets (no trimming)
      const end = page.getByLabel('Claim end offset');
      await end.fill('6');
      await end.press('Tab');
      how = 'offsets [0, 6)';
    }
    const readout = await page.getByTestId('selection-readout').innerText();
    if (!/chars \[\d+, \d+\)/.test(readout)) throw new Error(`no span selected: ${readout}`);
    await page.getByRole('button', { name: 'Investigate claim', exact: true }).click();
    await page.getByText('Claim under audit', { exact: true }).waitFor();
    const c = lastCall(/selflab\/cases$/);
    if (!c || c.status !== 200) throw new Error(`POST /cases → ${c && c.status}`);
    const txt = await bodyText();
    if (MODE === 'fixture' && !/\[DEMO\]/.test(txt)) throw new Error('case selector does not mark the case [DEMO]');
    return { how, readout: readout.slice(0, 120) };
  });

  if (caseOk) {
    await step('suggest-tests', async () => {
      await page.getByRole('button', { name: /suggest tests/i }).click();
      await page.getByRole('list', { name: /suggested tests/i }).waitFor();
      return { n: await page.getByRole('list', { name: /suggested tests/i }).locator('li').count() };
    }, { required: false });

    await step('compare-contexts-commit-run', async () => {
      await page.getByRole('button', { name: /^compare contexts/i }).click();
      await page.getByRole('button', { name: /commit spec/i }).click();
      await page.waitForTimeout(1500);
      const c = lastCall(/selflab\/experiments$/);
      if (!c || c.status !== 200) throw new Error(`commit refused: POST /experiments → ${c && c.status} ${c && c.body}`);
      await page.getByRole('button', { name: /^run$/i }).click();
      await page.getByText(/trials analysed/i).first().waitFor();
      return { committed: true };
    }, { required: false, knownDefect: /POST \/experiments → 500|replacement_text/ });
  }

  // ------------------------------------------------------------------ 4. battery (no case needed)
  let batteryId = null;
  let evidenceHiddenWithCase = false;
  const batteryOk = await step('battery-commit-run', async () => {
    await page.getByRole('button', { name: /source influence vs claimed source/i }).click();
    await page.getByLabel('Max trials').fill(MAX_TRIALS);
    await page.getByLabel('Max trials').press('Tab');
    await page.getByRole('button', { name: /commit spec/i }).click();
    await page.getByText(/plan_sha256/i).first().waitFor();
    const c = lastCall(/selflab\/experiments$/);
    if (!c || c.status !== 200) throw new Error(`POST /experiments → ${c && c.status}`);
    await page.getByRole('button', { name: /^run$/i }).click();
    await page.getByText(/trials analysed/i).first().waitFor();
    const c2 = lastCall(/selflab\/experiments$/);
    batteryId = ((c2 && c2.body.match(/"id":"(slx_[0-9a-f]+)"/)) || page.url().match(/exp=(slx_[0-9a-f]+)/) || [])[1] || null;
    const ev = page.locator('[aria-label^="Evidence: "]');
    await ev.first().waitFor({ timeout: 20000 }).catch(() => {});  // evidence loads after the analysis summary
    let n = await ev.count();
    if (!n) {  // with a case selected the evidence column is filtered to that case (see VALIDATION_REPORT D-UI2)
      evidenceHiddenWithCase = true;
      await page.goto(WEB + '/app/selflab' + (batteryId ? `?exp=${batteryId}` : ''), { waitUntil: 'networkidle' });
      await ev.first().waitFor({ timeout: 60000 }).catch(() => {});
      n = await ev.count();
    }
    if (!n) throw new Error('no evidence cards after the run');
    const titles = await ev.evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
    if (MODE === 'fixture' && !titles.every((t) => /\[FIXTURE\]/.test(t))) throw new Error('evidence titles not all marked [FIXTURE]');
    if (MODE === 'real' && titles.some((t) => /\[FIXTURE\]/.test(t))) throw new Error('real mode evidence marked [FIXTURE]');
    const statuses = (await bodyText()).match(/STATUS\s+(SUPPORTED|CONTRADICTED|INCONCLUSIVE|CONTROL FAILED|UNTESTED)/g) || [];
    return { n_evidence: n, statuses: statuses.slice(0, 12) };
  });

  await step('evidence-visible-while-a-case-is-selected', async () => {
    if (evidenceHiddenWithCase) throw new Error('evidence of the just-run experiment is hidden while a case is selected');
    return { ok: true };
  }, { required: false, knownDefect: /hidden while a case is selected/ });

  if (batteryOk) {
    await step('inspect-trial', async () => {
      if (!(await page.getByRole('button', { name: '#00000' }).count())) {
        const tab = page.getByRole('tab', { name: /trials/i });
        if (await tab.count()) await tab.first().click();
      }
      await page.getByRole('button', { name: '#00000' }).first().click();
      await page.getByText(/ground truth \(separate fields\)/i).waitFor();
      return { ok: true };
    });
    await step('open-in-lab-replay', async () => {
      await page.getByRole('button', { name: /^open in lab$/i }).first().click();
      for (let i = 0; i < 120 && !lastCall(/open-in-lab$/); i++) await page.waitForTimeout(250);
      const c = lastCall(/open-in-lab$/);
      if (!c) throw new Error('no open-in-lab request was made');
      if (c.status !== 200) throw new Error(`open-in-lab → ${c.status} ${c.body}`);
      const reproduced = /"reproduced":true/.test(c.body);
      await page.waitForURL(/\/app\?run=/, { timeout: 30000 });  // OPEN IN LAB navigates to the neural debugger
      await page.locator('canvas').first().waitFor();
      if (!reproduced) throw new Error(`replay did not reproduce the stored output: ${c.body.slice(0, 300)}`);
      return { status: c.status, reproduced, lab_url: page.url() };
    }, { required: false, knownDefect: /open-in-lab → 409/ });
    if (!page.url().includes('/app/selflab')) { await page.goBack(); await page.waitForLoadState('networkidle'); }
  }

  await step('open-chat-lab-run', async () => {  // the chat turn's own Lab run (messages-based) renders in /app?run=
    const chat = calls.find((c) => /selflab\/chat$/.test(c.path) && c.status === 200);
    const runId = chat && (chat.body.match(/"run_id":"(run_[0-9a-f]+)"/) || [])[1];
    if (!runId) throw new Error('no chat run id recorded');
    await page.goto(WEB + `/app?run=${runId}`, { waitUntil: 'networkidle' });
    await page.locator('canvas').first().waitFor();
    const txt = await bodyText();
    if (MODE === 'fixture' && !/DEMO/i.test(txt)) throw new Error('Lab view of a fixture run is not marked DEMO');
    return { url: page.url() };
  });

  if (batteryOk) {
    await step('export-preview', async () => {
      if (batteryId) await page.goto(WEB + `/app/selflab?exp=${batteryId}`, { waitUntil: 'networkidle' });
      const prev = page.getByRole('button', { name: /export evidence: preview/i });
      const tab = page.getByRole('tab', { name: /export/i });
      await tab.first().waitFor({ timeout: 30000 }).catch(() => {});
      if (!(await prev.count()) && (await tab.count())) await tab.first().click();
      await prev.first().click();
      const table = page.getByRole('table', { name: /fields included in the export/i });
      await table.waitFor();
      const txt = await table.innerText();
      return { rows: (await table.locator('tr').count()), flags_free_text: /free/i.test(await bodyText()), sample: txt.slice(0, 160) };
    });
  }

  await step('no-page-errors', async () => {
    if (pageErrors.length) throw new Error(pageErrors.join(' | ').slice(0, 600));
    return { n: 0 };
  });

  await browser.close();
  const failed = results.filter((r) => r.status === 'fail' && r.required);
  const known = results.filter((r) => r.status === 'known_defect');
  const summary = { mode: MODE, web: WEB, api: API, started: new Date(t0).toISOString(), seconds: (Date.now() - t0) / 1000,
                    passed: results.filter((r) => r.status === 'pass').length, failed_required: failed.length,
                    failed_optional: results.filter((r) => r.status === 'fail' && !r.required).length,
                    known_defects: known.length, steps: results, selflab_calls: calls };
  fs.writeFileSync(path.join(OUT_DIR, 'summary.json'), JSON.stringify(summary, null, 2));
  lap(`summary: ${summary.passed} pass, ${failed.length} required fail, ${summary.failed_optional} optional fail, ${known.length} known defect → ${path.join(OUT_DIR, 'summary.json')}`);
  process.exit(failed.length || (STRICT && known.length) ? 1 : 0);
})().catch((e) => { console.error('SMOKE CRASHED', e); process.exit(2); });
