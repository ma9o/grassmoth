import type { CoefficientRole, ConstructId, Expression, ParameterId } from "@grassmoth/api-types";
import { jStat } from "jstat";
import { assertNever } from "@/lib/assert-never";

/** A display cannot invent a missing operand or substitute another probability law. */
export class PlotUnavailable extends Error {}

export type ExpressionValue = number | readonly number[];

/** One draw's scientific values, after applying its declared parameter transforms. */
export interface ExpressionInputs {
  readonly state: (id: ConstructId) => number;
  readonly parameter: (id: ParameterId, role?: CoefficientRole) => ExpressionValue;
}

export function scalarValue(value: ExpressionValue): number {
  if (typeof value !== "number")
    throw new PlotUnavailable("This expression needs a scalar coordinate.");
  return value;
}

export function finiteValue(value: number): number {
  if (!Number.isFinite(value))
    throw new PlotUnavailable("The expression is undefined for some of these input draws.");
  return value;
}

const vector = (value: ExpressionValue): readonly number[] =>
  typeof value === "number" ? [value] : value;

function combine(
  left: ExpressionValue,
  right: ExpressionValue,
  operation: (a: number, b: number) => number,
): ExpressionValue {
  if (typeof left === "number" && typeof right === "number") return operation(left, right);
  const first = vector(left);
  const second = vector(right);
  const length = Math.max(first.length, second.length);
  if (
    (first.length !== 1 && first.length !== length) ||
    (second.length !== 1 && second.length !== length)
  )
    throw new PlotUnavailable("The expression's vector coordinates do not align.");
  return Array.from({ length }, (_, index) => {
    const a = first[first.length === 1 ? 0 : index];
    const b = second[second.length === 1 ? 0 : index];
    if (a === undefined || b === undefined) throw new PlotUnavailable("Missing vector coordinate.");
    return operation(a, b);
  });
}

export const sigmoid = (value: number) =>
  value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));

/** Interpret the saved syntax tree; no generated code, fitting, or trajectory solver. */
export function evaluateExpression(
  expression: Expression,
  inputs: ExpressionInputs,
): ExpressionValue {
  const evaluate = (value: Expression) => evaluateExpression(value, inputs);
  switch (expression.kind) {
    case "literal":
      return expression.value;
    case "state":
      return inputs.state(expression.construct_id);
    case "coefficient":
      if (expression.value === null)
        throw new PlotUnavailable("A coefficient has no value or law yet.");
      return typeof expression.value === "number"
        ? expression.value
        : inputs.parameter(expression.value, expression.role);
    case "binary": {
      const left = evaluate(expression.left);
      const right = evaluate(expression.right);
      switch (expression.operator) {
        case "add":
          return combine(left, right, (a, b) => a + b);
        case "subtract":
          return combine(left, right, (a, b) => a - b);
        case "multiply":
          return combine(left, right, (a, b) => a * b);
        case "divide":
          return combine(left, right, (a, b) => a / b);
        case "power":
          return combine(left, right, (a, b) => a ** b);
        case "maximum":
          return combine(left, right, Math.max);
        default:
          return assertNever(expression.operator);
      }
    }
    case "call": {
      const args = expression.arguments.map(evaluate);
      const at = (index: number) => {
        const value = args[index];
        if (value === undefined) throw new Error("Saved expression is missing an argument.");
        return value;
      };
      switch (expression.function) {
        case "exp":
          return Math.exp(scalarValue(at(0)));
        case "sigmoid":
          return sigmoid(scalarValue(at(0)));
        case "normal_cdf":
          return jStat.normal.cdf(scalarValue(at(0)), 0, 1);
        case "ordered_cutpoints": {
          let cutpoint = scalarValue(at(0));
          return [cutpoint, ...vector(at(1)).map((gap) => (cutpoint += gap))];
        }
        case "category_logits":
          return [
            0,
            ...vector(
              combine(
                at(1),
                combine(at(2), at(0), (a, b) => a * b),
                (a, b) => a + b,
              ),
            ),
          ];
        default:
          return assertNever(expression.function);
      }
    }
    default:
      return assertNever(expression);
  }
}

/** Exact forward differentiation of a scalar potential, on the same draw as its value. */
export function potentialDrift(
  expression: Expression,
  inputs: ExpressionInputs,
  target: ConstructId,
): number {
  const dual = (node: Expression): readonly [number, number] => {
    if (node.kind === "state")
      return [inputs.state(node.construct_id), Number(node.construct_id === target)];
    if (node.kind === "literal" || node.kind === "coefficient")
      return [scalarValue(evaluateExpression(node, inputs)), 0];
    if (node.kind === "binary") {
      const [a, da] = dual(node.left);
      const [b, db] = dual(node.right);
      switch (node.operator) {
        case "add":
          return [a + b, da + db];
        case "subtract":
          return [a - b, da - db];
        case "multiply":
          return [a * b, da * b + a * db];
        case "divide":
          return [a / b, (da * b - a * db) / (b * b)];
        case "power":
          return [
            a ** b,
            (da === 0 || b === 0 ? 0 : b * a ** (b - 1) * da) +
              (db === 0 ? 0 : a ** b * Math.log(a) * db),
          ];
        case "maximum":
          return [Math.max(a, b), a === b ? (da + db) / 2 : a > b ? da : db];
        default:
          return assertNever(node.operator);
      }
    }
    const arg = node.arguments[0];
    if (!arg) throw new Error("Saved expression is missing an argument.");
    const [a, da] = dual(arg);
    switch (node.function) {
      case "exp":
        return [Math.exp(a), Math.exp(a) * da];
      case "sigmoid": {
        const value = sigmoid(a);
        return [value, value * (1 - value) * da];
      }
      case "normal_cdf":
        return [jStat.normal.cdf(a, 0, 1), jStat.normal.pdf(a, 0, 1) * da];
      case "ordered_cutpoints":
      case "category_logits":
        throw new PlotUnavailable("A potential must be scalar.");
      default:
        return assertNever(node.function);
    }
  };
  return -dual(expression)[1];
}
