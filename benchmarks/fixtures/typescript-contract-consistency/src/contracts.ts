export type AccountState = "active" | "disabled";

export interface Account {
  id: string;
  email: string;
  displayName: string;
  state: AccountState;
}
