import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import test from 'node:test';

const appPath = new URL('../src/App.tsx', import.meta.url);
const indexPath = new URL('../index.html', import.meta.url);
const serverPath = new URL('../local/server.mjs', import.meta.url);
const stylePath = new URL('../src/style.css', import.meta.url);
const historyModulePath = new URL('../src/chat-history.js', import.meta.url);
const workspaceSettingsPath = new URL('../src/workspace-settings.js', import.meta.url);

test('the frontend stays a simple chat rather than a model dashboard', async () => {
  const [app, html, server, css, historyModule, workspaceSettings] = await Promise.all([
    fs.readFile(appPath, 'utf8'),
    fs.readFile(indexPath, 'utf8'),
    fs.readFile(serverPath, 'utf8'),
    fs.readFile(stylePath, 'utf8'),
    fs.readFile(historyModulePath, 'utf8'),
    fs.readFile(workspaceSettingsPath, 'utf8'),
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
  assert.match(app, /content\.toLowerCase\(\) === '\/clear'/);
  assert.match(app, /job\.partialReply/);
  assert.match(app, /within 15 seconds/);
  assert.match(app, /streamingReply/);
  assert.match(app, /aria-label="Conversation"/);
  assert.match(app, /aria-live="off"/);
  assert.match(app, /SYNTRA response/);
  assert.match(app, /prefers-reduced-motion: reduce/);
  assert.match(css, /\.chat-app \{[^}]*min-height: 0/s);
  assert.match(css, /\.chat-composer textarea \{ font-size: 16px; \}/);
  assert.match(app, /shouldFollowRef\.current/);
  assert.match(app, /onScroll=\{handleThreadScroll\}/);
  assert.match(app, /distanceFromBottom < 120/);
  assert.match(app, /loadSessionState\(\)/);
  assert.match(app, /saveSessionState\(sessionState\)/);
  assert.match(app, /MAX_VISIBLE_MESSAGES = 60/);
  assert.match(app, /aria-label="Conversation history"/);
  assert.match(app, /Search conversations/);
  assert.match(app, /aria-label="New chat"/);
  assert.match(app, /Rename/);
  assert.match(app, /Delete/);
  assert.match(app, /Saved locally on this device/);
  assert.match(app, /selectSession\(session.id\)/);
  assert.match(app, /saveRename/);
  assert.match(app, /aria-label="Active workspace"/);
  assert.match(app, /\/local-api\/workspace\/status/);
  assert.match(app, /\/local-api\/workspace\/switch/);
  assert.match(app, /submitWorkspaceSwitch/);
  assert.match(app, /Review saved proposals in chat/);
  assert.match(app, /loadWorkspacePath\(\)/);
  assert.match(workspaceSettings, /WORKSPACE_PATH_KEY/);
  assert.match(css, /\.workspace-popover/);
  assert.match(app, /role: 'user' as const/);
  assert.match(app, /role: 'assistant' as const/);
  assert.match(app, /Save the user's prompt immediately too/);
  assert.match(app, /slice\(-MAX_VISIBLE_MESSAGES\)/);
  assert.match(historyModule, /MAX_TOTAL_CHARS/);
  assert.match(app, /message\.role === 'assistant'/);
  assert.match(app, /Response interrupted before completion/);
  assert.match(css, /\.user-message-text \{[^}]*white-space: pre-wrap/s);
  assert.match(server, /phase: "streaming"/);
  assert.match(server, /onToken: \(_delta, fullText\)/);
  assert.match(server, /partialReply, updatedAt/);
  assert.match(server, /new task must never inherit proposals/);
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
  ]) {
    assert.equal(frontend.toLowerCase().includes(label.toLowerCase()), false, `legacy dashboard label leaked into frontend: ${label}`);
  }
});
