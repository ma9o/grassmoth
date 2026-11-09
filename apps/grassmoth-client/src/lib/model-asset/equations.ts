import type {
  BinaryOperator,
  CallExpression,
  CausalEdgeSpec,
  ConstructId,
  ConstructSpec,
  DynamicsMechanismSpec,
  Expression,
  IndicatorSpec,
  ParameterSpec,
} from "@grassmoth/api-types";
import type { ModelEntities } from "./entities";
import { observationArguments } from "./observation-law";

const TEXT_ESCAPES: Readonly<Record<string, string>> = {
  "\\": String.raw`\textbackslash{}`,
  "{": String.raw`\{`,
  "}": String.raw`\}`,
  "%": String.raw`\%`,
  "&": String.raw`\&`,
  "#": String.raw`\#`,
  $: String.raw`\$`,
  _: " ",
  "^": String.raw`\textasciicircum{}`,
  "~": String.raw`\textasciitilde{}`,
};
const FUNCTIONS: Record<CallExpression["function"], string> = {
  exp: String.raw`\exp`,
  sigmoid: String.raw`\operatorname{logistic}`,
  normal_cdf: String.raw`\Phi`,
  ordered_cutpoints: String.raw`\operatorname{ordered\_cutpoints}`,
  category_logits: String.raw`\operatorname{category\_logits}`,
};

function text(label: string): string {
  return String.raw`\text{${label.replaceAll(/[\\{}%&#$_^~]/g, (character) => TEXT_ESCAPES[character] ?? character)}}`;
}

function state(entities: ModelEntities, id: ConstructId): string {
  const construct = entities.constructById.get(id);
  if (!construct) throw new Error(`Equation references absent construct ${id}`);
  return String.raw`\eta_{${text(construct.name)}}(t)`;
}

function parameterLatex(parameter: ParameterSpec): string {
  const theta = String.raw`\theta_{${text(parameter.name)}}`;
  const interval = String.raw`\Delta_{${text(parameter.name)}}`;
  switch (parameter.transform.kind) {
    case "dt_persistence_to_ct_decay": {
      return String.raw`\frac{-\log\left(${theta}\right)}{${interval}}`;
    }
    case "dt_effect_to_ct_rate": {
      return String.raw`\frac{${theta}}{${interval}}`;
    }
    case "identity":
    case "initial_state_correlation": {
      return theta;
    }
  }
}

function binaryLatex(operator: BinaryOperator, left: string, right: string): string {
  switch (operator) {
    case "add": {
      return String.raw`\left(${left} + ${right}\right)`;
    }
    case "subtract": {
      return String.raw`\left(${left} - ${right}\right)`;
    }
    case "multiply": {
      return String.raw`\left(${left} \cdot ${right}\right)`;
    }
    case "divide": {
      return String.raw`\frac{${left}}{${right}}`;
    }
    case "power": {
      return String.raw`{\left(${left}\right)}^{${right}}`;
    }
    case "maximum": {
      return String.raw`\max\left(${left},\;${right}\right)`;
    }
  }
}

/** Typeset the saved syntax tree without simplifying or evaluating it. */
function expressionLatex(entities: ModelEntities, expression: Expression): string {
  const render = (operand: Expression) => expressionLatex(entities, operand);
  switch (expression.kind) {
    case "literal": {
      return String(expression.value).replace(/e([+-]?\d+)/, String.raw`\times 10^{$1}`);
    }
    case "state": {
      return state(entities, expression.construct_id);
    }
    case "coefficient": {
      if (expression.value === null) return String.raw`\underbrace{?}_{${text(expression.role)}}`;
      if (typeof expression.value === "number") {
        return render({ kind: "literal", value: expression.value });
      }
      const parameter = entities.parameterById.get(expression.value);
      if (!parameter) throw new Error(`Equation references absent parameter ${expression.value}`);
      return parameterLatex(parameter);
    }
    case "binary": {
      const left = render(expression.left);
      const right = render(expression.right);
      return binaryLatex(expression.operator, left, right);
    }
    case "call": {
      return String.raw`${FUNCTIONS[expression.function]}\left(${expression.arguments.map((argument) => render(argument)).join(String.raw`,\;`)}\right)`;
    }
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
    .map(([name, value]) => String.raw`\mathrm{${name}}=${expressionLatex(entities, value)}`)
    .join(String.raw`,\; `);
  return String.raw`y_{${text(indicator.observation.name)}}(t) \sim \operatorname{${law.distribution}}\left(${argumentsLatex}\right)`;
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
        ? String.raw`-\frac{\partial}{\partial ${state(entities, target)}}\left[${expression}\right]`
        : expression;
    })
    .join(" + ");
}

/** The local contribution that this edge adds to the target construct's drift. */
export function edgeEquation(edge: CausalEdgeSpec, entities: ModelEntities): string | null {
  return edge.mechanisms.length === 0
    ? null
    : String.raw`g(\eta,t) = ${driftTerms(edge.mechanisms, edge.effect.id, entities)}`;
}

/** Display authored mechanisms or direct DAG links, never a newly inferred noise projection. */
export function constructEquation(construct: ConstructSpec, entities: ModelEntities) {
  if (construct.temporal_status === "time_invariant") {
    return {
      title: "Static state",
      latex: String.raw`\mathrm{d}${state(entities, construct.id)} = 0`,
    };
  }
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
          latex: String.raw`${state(entities, construct.id)} \to \left\{${children.map((edge) => state(entities, edge.effect.id)).join(String.raw`,\;`)}\right\}`,
        };
  }
  return {
    title: "Authored drift",
    latex: String.raw`f_{${text(construct.name)}}(\eta,t) = ${driftTerms(mechanisms, construct.id, entities)}`,
  };
}
