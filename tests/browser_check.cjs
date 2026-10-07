// Browser-level acceptance checks. Run with Playwright available in NODE_PATH.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
let browser;
let page;

(async () => {
  browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || undefined, headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 1500 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(process.env.BASE_URL || 'http://127.0.0.1:8501');
  await page.getByRole('heading', { name: 'Inspection Lab', exact: true }).waitFor({ timeout: 60000 });
  await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
  await page.getByText('Above threshold', { exact: true }).waitFor({ timeout: 60000 });
  await page.getByRole('button', { name: 'Download overlay', exact: true }).waitFor();
  await page.getByText('Ground truth · cyan annotation', { exact: true }).waitFor();
  const scoreBefore = await page.locator('[data-testid="stMetric"]').filter({ hasText: 'Raw anomaly score' }).innerText();
  await page.getByText('Fixed calibration', { exact: true }).click();
  await page.getByText('Calibration scale shared across images', { exact: false }).waitFor();
  await page.getByText('Per-image maximum', { exact: true }).click();
  await page.getByText('Colors apply to this image only.', { exact: false }).waitFor();
  const scoreAfter = await page.locator('[data-testid="stMetric"]').filter({ hasText: 'Raw anomaly score' }).innerText();
  if (scoreBefore !== scoreAfter) throw new Error('Display scale changed score');
  await page.waitForFunction(() => [...document.querySelectorAll('summary')].filter(el => el.textContent.includes('Raw anomaly map and numeric color scale')).length === 1);
  await page.getByText('Raw anomaly map and numeric color scale', { exact: true }).click();
  await page.getByRole('button', { name: 'Download raw map', exact: true }).waitFor();
  await page.getByText('Raw anomaly map and numeric color scale', { exact: true }).click();
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Download overlay', exact: true }).click()
  ]);
  if (download.suggestedFilename() !== 'inspection_overlay.png') throw new Error('Unexpected download filename');
  if (await download.failure()) throw new Error('Overlay download failed');
  await page.getByText('Above threshold', { exact: true }).waitFor();
  await page.locator('[data-testid="stMain"]').evaluate(el => el.scrollTo(0, 0));
  await page.screenshot({ path: 'artifacts/inspection-preview.png', fullPage: true, animations: 'disabled' });
  await page.getByRole('tab', { name: 'Experiments & failures' }).click();
  await page.getByRole('heading', { name: 'Reference budget' }).waitFor();
  await page.locator('[data-testid="stMain"]').evaluate(el => el.scrollTo(0, 0));
  await page.screenshot({ path: 'artifacts/experiments-preview.png', fullPage: true, animations: 'disabled' });
  await page.getByRole('tab', { name: 'Method & provenance' }).click();
  await page.getByRole('button', { name: 'Download English report' }).waitFor();
  if (process.env.CHECK_V2 !== '0') {
    await page.getByRole('button', { name: 'Verify artifact integrity' }).click();
    await page.getByText('All recorded artifact hashes match.', { exact: true }).waitFor({ timeout: 60000 });
    await page.getByRole('tab', { name: 'Repeated-seed benchmark' }).click();
    await page.getByRole('heading', { name: 'Stability across normal-reference samples' }).waitFor();
    await page.getByRole('button', { name: 'Download benchmark CSV' }).waitFor();
    await page.locator('[data-testid="stMain"]').evaluate(el => el.scrollTo(0, 0));
    await page.screenshot({ path: 'artifacts/benchmark-preview.png', fullPage: true, animations: 'disabled' });
  }
  await page.getByRole('tab', { name: 'Inspect an image' }).click();
  await page.getByText('Above threshold', { exact: true }).waitFor();
  await page.getByText('Upload image', { exact: true }).click();
  await page.getByText('Above threshold', { exact: true }).waitFor({ state: 'hidden', timeout: 30000 });
  await page.locator('input[type="file"]').setInputFiles({ name: 'invalid.png', mimeType: 'image/png', buffer: Buffer.from('not an image') });
  await page.getByText('Cannot read image:', { exact: false }).waitFor();
  if (await page.getByRole('button', { name: 'Run inspection', exact: true }).isEnabled()) throw new Error('Invalid image can be inspected');
  const dataRoot = process.env.INSPECTION_DATA_ROOT || path.resolve('data/mvtec_ad');
  await page.locator('input[type="file"]').setInputFiles(path.join(dataRoot, 'bottle/test/good/000.png'));
  await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
  await page.getByText('Within threshold', { exact: true }).waitFor({ timeout: 60000 });
  await page.getByText('Dataset example', { exact: true }).click();
  await page.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Product category' }).getByRole('combobox').click();
  await page.getByRole('option', { name: 'Metal Nut', exact: true }).click();
  await page.getByText('MVTec AD / metal_nut', { exact: false }).waitFor();
  await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
  await page.getByRole('button', { name: 'Download result JSON', exact: true }).waitFor({ timeout: 60000 });
  const [resultFile] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Download result JSON', exact: true }).click()
  ]);
  const result = JSON.parse(fs.readFileSync(await resultFile.path(), 'utf8'));
  if (result.overlay_style !== 'Response weighted' || result.opacity_exponent !== 3) throw new Error('Overlay provenance missing');
  if (result.category !== 'metal_nut' || !result.run.startsWith('metal_nut_')) throw new Error('Wrong category model');
  await page.getByRole('tab', { name: 'Repeated-seed benchmark' }).click();
  await page.getByRole('heading', { name: 'Stability across normal-reference samples' }).waitFor();
  await page.getByRole('tab', { name: 'Controlled comparison' }).click();
  await page.getByRole('heading', { name: 'Algorithm and memory-capacity comparison' }).waitFor({ timeout: 60000 });
  await page.getByRole('button', { name: 'Download controlled comparison' }).waitFor();
  await page.getByRole('tab', { name: 'Inspect an image' }).click();
  await page.screenshot({ path: 'artifacts/metal-nut-preview.png', fullPage: true, animations: 'disabled' });
  const runSelector = page.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Completed run' }).getByRole('combobox');
  await runSelector.fill('controlled_v4_metal_nut_padim_seed42');
  await page.getByRole('option', { name: 'controlled_v4_metal_nut_padim_seed42', exact: true }).click();
  await page.getByRole('button', { name: 'Download result JSON', exact: true }).waitFor({ state: 'hidden' });
  await page.getByRole('heading', { name: 'Ready to inspect', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
  await page.getByRole('button', { name: 'Download result JSON', exact: true }).waitFor({ timeout: 60000 });
  const [padimFile] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Download result JSON', exact: true }).click()
  ]);
  const padimResult = JSON.parse(fs.readFileSync(await padimFile.path(), 'utf8'));
  if (padimResult.method !== 'padim') throw new Error('PaDiM adapter was not used');
  if (await page.locator('[data-testid="stException"]').count()) throw new Error('Streamlit exception shown');
  if (errors.length) throw new Error(errors.join('\n'));
  fs.writeFileSync('artifacts/browser-check.json', JSON.stringify({
    passed: true,
    checks: ['PaDiM checkpoint inference', 'controlled comparison tab', 'category switching and model identity', 'display scale preserves score', 'raw map export available', 'defect inference', 'ground-truth overlay', 'overlay download', 'persistent results', 'stale result invalidation', 'invalid upload rejection', 'experiment tab', 'report tab', 'normal upload', ...(process.env.CHECK_V2 !== '0' ? ['artifact verification', 'repeated-seed benchmark'] : []), 'no browser exceptions'],
    checked_at: new Date().toISOString()
  }, null, 2));
  await browser.close();
  console.log('Browser acceptance checks passed.');
})().catch(async error => {
  console.error(error);
  if (page) {
    console.error((await page.locator('body').innerText()).slice(0, 9000));
    await page.screenshot({ path: 'artifacts/browser-failure.png', fullPage: true, animations: 'disabled' }).catch(() => {});
  }
  if (browser) await browser.close();
  process.exit(1);
});
