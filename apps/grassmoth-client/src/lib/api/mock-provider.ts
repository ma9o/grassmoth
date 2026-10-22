export function isMockMode(): boolean {
  const v = process.env.NEXT_PUBLIC_MOCK_DATA;
  return !!v && v !== "false";
}

/** Returns the fixture user ID, e.g. "DEFAULT". */
export function getMockFixture(): string {
  const v = process.env.NEXT_PUBLIC_MOCK_DATA;
  if (!v || v === "true") return "DEFAULT";
  return v.toUpperCase();
}
