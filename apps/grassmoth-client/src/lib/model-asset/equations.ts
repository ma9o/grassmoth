import type {
  CallExpression,
  CausalEdgeSpec,
  ConstructId,
  ConstructSpec,
  DynamicsMechanismSpec,
  Expression,
  IndicatorSpec,
} from "@grassmoth/api-types";
import { assertNever } from "@/lib/assert-never";
import type { ModelEntities } from "./entities";
import { observationArguments } from "./observation-law";

const TEXT_ESCAPES: Readonly<Record<string, string>> = {
  "\\": "\\textbackslash{}",
  "{": "\\{",
  "}": "\\}",
  "%": "\\%",
  "&": "\\&",
  "#": "\\#",
  $: "\\$",
  _: " ",
  "^": "\\textasciicircum{}",
  "~": "\\textasciitilde{}",
};
const FUNCTIONS: Record<CallExpression["function"], string> = {
  exp: "\\exp",
  sigmoid: "\\operatorname{logistic}",
  normal_cdf: "\\Phi",
  ordered_cutpoints: "\\operatorname{ordered\\_cutpoints}",
  category_logits: "\\operatorname{category\\_logits}",
};

function text(label: string): string {
  return `\\text{${label.replace(/[\\{}%&#$_^~]/g, (character) => TEXT_ESCAPES[character] ?? character)}}`;
}

function state(entities: ModelEntities, id: ConstructId): string {
  const construct = entities.constructById.get(id);
  if (!construct) throw new Error(`Equation references absent construct ${id}`);
  return `\\eta_{${text(construct.name)}}(t)`;
}

/** Typeset the saved syntax tree without simplifying or evaluating it. */
function expressionLatex(entities: ModelEntities, expression: Expression): string {
  const render = (operand: Expression) => expressionLatex(entities, operand);
  switch (expression.kind) {
    case "literal":
      return String(expression.value).replace(/e([+-]?\d+)/, "\\times 10^{$1}");
    case "state":
      return state(entities, expression.construct_id);
    case "coefficient": {
      if (expression.value === null) return `\\underbrace{?}_{${text(expression.role)}}`;
      if (typeof expression.value === "number")
        return render({ kind: "literal", value: expression.value });
      const parameter = entities.parameterById.get(expression.value);
      if (!parameter) throw new Error(`Equation references absent parameter ${expression.value}`);
      const theta = `\\theta_{${text(parameter.name)}}`;
      const interval = `\\Delta_{${text(parameter.name)}}`;
      switch (parameter.transform.kind) {
        case "dt_persistence_to_ct_decay":
          return `\\frac{-\\log\\left(${theta}\\right)}{${interval}}`;
        case "dt_effect_to_ct_rate":
          return `\\frac{${theta}}{${interval}}`;
        case "identity":
        case "initial_state_correlation":
          return theta;
        default:
          return assertNever(parameter.transform);
      }
    }
    case "binary": {
      const left = render(expression.left);
      const right = render(expression.right);
      const operator = expression.operator;
      switch (operator) {
        case "add":
          return `\\left(${left} + ${right}\\right)`;
        case "subtract":
          return `\\left(${left} - ${right}\\right)`;
        case "multiply":
          return `\\left(${left} \\cdot ${right}\\right)`;
        case "divide":
          return `\\frac{${left}}{${right}}`;
        case "power":
          return `{\\left(${left}\\right)}^{${right}}`;
        case "maximum":
          return `\\max\\left(${left},\\;${right}\\right)`;
        default:
          return assertNever(operator);
      }
    }
    case "call":
      return `${FUNCTIONS[expression.function]}\\left(${expression.arguments.map(render).join(",\\;")}\\right)`;
    default:
      return assertNever(expression);
  }
}

/** The authored native law, including its explicitly unfinished operands. */
export function observationEquation(
  indicator: IndicatorSpec,
  entities: ModelEntities,
): string | null {
  if (!indicator.likelihood) return null;
  const law = indicator.likelihood.law;
  const argumentsLatex = observationArguments(law)
    .map(([name, value]) => `\\mathrm{${name}}=${expressionLatex(entities, value)}`)
    .join(",\\; ");
  return `y_{${text(indicator.observation.name)}}(t) \\sim \\operatorname{${law.distribution}}\\left(${argumentsLatex}\\right)`;
}

function driftTerms(
  mechanisms: readonly DynamicsMechanismSpec[],
  target: ConstructId,
  entities: ModelEntities,
): string {
  return mechanisms
    .map((mechanism) => {
      const expression = expressionLatex(entities, mechanism.expression);
      return mechanism.kind === "potential"
        ? `-\\frac{\\partial}{\\partial ${state(entities, target)}}\\left[${expression}\\right]`
        : expression;
    })
    .join(" + ");
}

/** The local contribution that this edge adds to the target construct's drift. */
export function edgeEquation(edge: CausalEdgeSpec, entities: ModelEntities): string | null {
  return edge.mechanisms.length === 0
    ? null
    : `g(\\eta,t) = ${driftTerms(edge.mechanisms, edge.effect.id, entities)}`;
}

/** Display authored mechanisms or direct DAG links, never a newly inferred noise projection. */
export function constructEquation(construct: ConstructSpec, entities: ModelEntities) {
  if (construct.temporal_status === "time_invariant")
    return {
      title: "Static state",
      latex: `\\mathrm{d}${state(entities, construct.id)} = 0`,
    };
  const mechanisms = [
    ...construct.dynamics,
    ...entities.edges
      .filter((edge) => edge.effect.id === construct.id)
      .flatMap((edge) => edge.mechanisms),
  ];
  if (mechanisms.length === 0) {
    const children = entities.edges.filter((edge) => edge.cause.id === construct.id);
    return children.length === 0
      ? null
      : {
          title: "Authored connections",
          latex: `${state(entities, construct.id)} \\to \\left\\{${children.map((edge) => state(entities, edge.effect.id)).join(",\\;")}\\right\\}`,
        };
  }
  return {
    title: "Authored drift",
    latex: `f_{${text(construct.name)}}(\\eta,t) = ${driftTerms(mechanisms, construct.id, entities)}`,
  };
}
