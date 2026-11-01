import { Account as UserAccount, AccountState as UserAccountState } from "./contracts";

export function normalizeAccount(account: UserAccount): UserAccount {
  const state: UserAccountState = account.state;
  return {
    id: account.id,
    email: account.email.trim(),
    displayName: account.displayName.trim(),
    status: state
  };
}
