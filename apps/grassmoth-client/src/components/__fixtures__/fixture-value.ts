/** A required part of an authored test/story fixture fails at its loading boundary. */
export function fixtureValue<T>(value: T | null | undefined): T {
  if (value == null) throw new Error("Fixture is missing a required value");
  return value;
}
