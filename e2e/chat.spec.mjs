import { test, expect } from "@playwright/test";

const API_TOKEN = "syntra-e2e-session-token-" + "x".repeat(48);
const WORKSPACE = "C:\\SYNTRA-E2E";

async function mockLocalApi(page) {
  await page.route("**/local-api/**", async route => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    const method = request.method();
    const json = payload => route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(payload)
    });

    if (pathname === "/local-api/session") {
      return json({ token: API_TOKEN });
    }
    if (pathname === "/local-api/workspace/status") {
      return json({ root: WORKSPACE, pendingProposals: [], staleProposals: [] });
    }
    if (pathname === "/local-api/agent/runs") {
      return json({ runs: [] });
    }
    if (pathname === "/local-api/chat" && method === "POST") {
      const body = request.postDataJSON();
      return json({
        jobId: "e2e-chat-job",
        status: "completed",
        reply: "Browser integration response: " + body.message
      });
    }
    if (pathname === "/local-api/workspace/exclusions") {
      return json({ root: WORKSPACE, patterns: [], updatedAt: null });
    }
    if (pathname === "/local-api/tools/execute") {
      return json({ ok: true, output: { notes: [] } });
    }
    if (pathname === "/local-api/approvals") {
      return json({ approvalToken: "e2e-approval-token" });
    }
    return json({ ok: true, output: {} });
  });
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.clear();
    sessionStorage.clear();
  });
  await mockLocalApi(page);
});

test("chat submission renders the user's message and the completed local response", async ({ page }) => {
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible();

  const prompt = "Explain how the repository graph resolves imports";
  await page.getByRole("textbox", { name: "Message SYNTRA" }).fill(prompt);
  await page.getByRole("button", { name: "Send message" }).click();

  await expect(page.getByRole("article", { name: "Your message" })).toContainText(prompt);
  await expect(page.getByRole("article", { name: "SYNTRA response" }))
    .toContainText("Browser integration response: " + prompt);
  expect(pageErrors).toEqual([]);
});

test("conversation history supports rename and starting a fresh chat", async ({ page }) => {
  await page.goto("/");
  const prompt = "Investigate parser behavior";
  await page.getByRole("textbox", { name: "Message SYNTRA" }).fill(prompt);
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByRole("article", { name: "SYNTRA response" })).toBeVisible();

  await page.getByRole("button", { name: "Open conversation history" }).click();
  const history = page.getByRole("complementary", { name: "Conversation history" });
  await expect(history).toBeVisible();

  await history.getByRole("button", { name: "Rename " + prompt }).click();
  await history.getByRole("textbox", { name: "Conversation name" }).fill("Parser investigation");
  await history.getByRole("button", { name: "Save conversation name" }).click();
  await expect(history.getByText("Parser investigation", { exact: true })).toBeVisible();

  await history.getByRole("button", { name: "New chat" }).click();
  await expect(page.getByRole("heading", { name: "How can I help?" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open conversation history" })).toBeVisible();
});
