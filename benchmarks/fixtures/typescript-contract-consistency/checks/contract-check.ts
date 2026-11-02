import type { Account } from "../src/contracts";
import { normalizeAccount } from "../src/normalizer";

const source: Account = { id: "acct-1", email: "user@example.test", displayName: "User", state: "active" };
const normalized: Account = normalizeAccount(source);
void normalized;
