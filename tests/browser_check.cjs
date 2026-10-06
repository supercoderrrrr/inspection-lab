// Run against a local server with calibrated baseline/PatchCore fits and the bottle dataset.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const data = process.env.INSPECTION_DATA_ROOT || path.join('data', 'mvtec_ad');
const goodImage = path.join(data, 'bottle', 'test', 'good', '000.png');
const output = path.resolve('artifacts');
fs.mkdirSync(output, { recursive: true });

function expected(method, relative) {
  const csv = fs.readFileSync(path.join('results', `bottle-${method}`, 'predictions.csv'), 'utf8');
  const row = csv.split(/\r?\n/).slice(1).map(line => line.split(',')).find(row => row[0] === relative);
  assert(row, `Missing public prediction: ${relative}`);
  return { score: Number(row[4]), decision: Boolean(Number(row[5])), sha256: row[1] };
}

(async () => {
  assert(fs.existsSync(goodImage), 'Set INSPECTION_DATA_ROOT to the MVTec AD parent folder.');
  const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || undefined, headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    page.setDefaultTimeout(30000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const waitForResult = () => page.getByRole('button', { name: 'Download result', exact: true }).waitFor();
    async function download(name) {
      const [result] = await Promise.all([
        page.waitForEvent('download'),
        page.getByRole('button', { name, exact: true }).click()
      ]);
      assert.equal(await result.failure(), null);
      return { name: result.suggestedFilename(), bytes: fs.readFileSync(await result.path()) };
    }
    async function verifyResult(method, relative) {
      const reference = expected(method, relative);
      await page.locator('[data-testid="stMetric"]').filter({ hasText: 'Raw anomaly score' })
        .getByText(reference.score.toFixed(4), { exact: true }).waitFor();
      const artifact = await download('Download result');
      const result = JSON.parse(artifact.bytes.toString());
      assert.equal(result.method, method);
      assert.equal(result.image_sha256, reference.sha256);
      assert.equal(result.decision, reference.decision);
      assert(Math.abs(result.raw_anomaly_score - reference.score) < 1e-5, 'UI score differs from evaluated CLI score.');
      return result;
    }
    async function chooseExample(name) {
      const selector = page.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Example' }).getByRole('combobox');
      await selector.click();
      await selector.fill(name);
      await page.getByRole('option', { name, exact: true }).click();
    }
    async function chooseModel(name) {
      await page.locator('[data-testid="stSelectbox"]').filter({ hasText: 'Saved model' }).getByRole('combobox').click();
      await page.getByRole('option', { name, exact: true }).click();
    }
    async function waitForClearedResult() {
      await page.waitForFunction(() => ![...document.querySelectorAll('[data-testid="stMetric"]')]
        .some(element => element.textContent.includes('Raw anomaly score')));
    }

    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:8501');
    await page.getByRole('heading', { name: 'Inspection Lab', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
    await waitForResult();
    await verifyResult('patchcore', 'test/broken_large/000.png');
    const overlay = await download('Download overlay');
    assert.equal(overlay.name, 'inspection_overlay.png');
    assert.equal(overlay.bytes.subarray(1, 4).toString(), 'PNG');
    const raw = await download('Download raw map');
    assert.equal(raw.name, 'anomaly_map.npy');
    assert.equal(raw.bytes.subarray(1, 6).toString(), 'NUMPY');
    await page.screenshot({ path: path.join(output, 'ui-inspection.png'), fullPage: true, animations: 'disabled' });

    await chooseExample('good / 000.png');
    await waitForClearedResult();
    await page.getByText('Ready to inspect. The saved model and its fixed threshold determine the decision.', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
    await waitForResult();
    await verifyResult('patchcore', 'test/good/000.png');
    await page.getByText('Upload image', { exact: true }).click();
    await waitForClearedResult();
    await page.locator('input[type=file]').setInputFiles(goodImage);
    await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
    await waitForResult();
    await verifyResult('patchcore', 'test/good/000.png');
    assert.equal(await page.getByText('Ground truth · cyan annotation', { exact: true }).count(), 0);
    await page.locator('input[type=file]').setInputFiles({ name: 'corrupt.png', mimeType: 'image/png', buffer: Buffer.from('invalid') });
    await page.getByText('The uploaded file is not a readable PNG or JPEG.', { exact: true }).waitFor();
    await waitForClearedResult();

    await page.locator('input[type=file]').setInputFiles(goodImage);
    await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
    await waitForResult();
    await chooseModel('baseline · baseline-calibrated');
    await waitForClearedResult();
    await page.getByText('Ready to inspect. The saved model and its fixed threshold determine the decision.', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Run inspection', exact: true }).click();
    await waitForResult();
    await verifyResult('baseline', 'test/good/000.png');

    await page.getByRole('tab', { name: 'Evaluation', exact: true }).click();
    await page.getByRole('heading', { name: 'Initial bottle comparison', exact: true }).waitFor();
    await page.getByText('patchcore: predictions and failures', { exact: true }).click();
    const csv = await download('Download predictions');
    assert.equal(csv.name, 'patchcore_predictions.csv');
    assert.equal(csv.bytes.toString().trim().split(/\r?\n/).length, 84);
    await page.screenshot({ path: path.join(output, 'ui-evaluation.png'), fullPage: true, animations: 'disabled' });
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser-check.json'), JSON.stringify({
      status: 'passed', checkedAt: new Date().toISOString(),
      checks: ['CLI/UI score agreement', 'example and model invalidation', 'upload equivalence',
        'corrupt upload rejection', 'overlay/raw/result downloads', 'public evaluation downloads']
    }, null, 2));
    process.stdout.write('Browser acceptance checks passed.\n');
  } finally {
    await browser.close();
  }
})().catch(error => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
