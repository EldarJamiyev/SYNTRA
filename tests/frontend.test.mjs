import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import test from 'node:test';

const appPath = new URL('../src/App.tsx', import.meta.url);
const indexPath = new URL('../index.html', import.meta.url);
const serverPath = new URL('../local/server.mjs', import.meta.url);

test('the frontend stays a simple chat rather than a model dashboard', async () => {
  const [app, html, server] = await Promise.all([
    fs.readFile(appPath, 'utf8'),
    fs.readFile(indexPath, 'utf8'),
    fs.readFile(serverPath, 'utf8'),
  ]);
  const frontend = app + '\n' + html;
  assert.match(app, /placeholder="Message SYNTRA"/);
  assert.match(app, /How can I help\?/);
  assert.match(app, /aria-label="Send message"/);
  assert.match(app, /from 'react-markdown'/);
  assert.match(app, /from 'remark-gfm'/);
  assert.match(app, /skipHtml/);
  assert.match(app, /SYNTRA's local service/);
  assert.match(app, /AbortSignal\.timeout\(15_000\)/);
  assert.match(app, /maxLength=\{12000\}/);
  assert.match(app, /setInput\(content\)/);
  assert.match(app, /within 15 seconds/);
  assert.match(app, /streamingReply/);
  assert.match(server, /phase: "streaming"/);
  assert.match(server, /onToken: \(_delta, fullText\)/);
  assert.match(app, /noopener noreferrer/);
  assert.match(html, /<title>SYNTRA Chat<\/title>/);
  for (const label of [
    'Studio Playground',
    'Coding Intelligence',
    'Build. Train. Generate.',
    'MODEL CHECKPOINT',
    'INFERENCE DEVICE',
    'TRAINING STEP',
    'CLI Commands',
    'Data & Training',
    'Workspace',
  ]) {
    assert.equal(frontend.toLowerCase().includes(label.toLowerCase()), false, `legacy dashboard label leaked into frontend: ${label}`);
  }
});
