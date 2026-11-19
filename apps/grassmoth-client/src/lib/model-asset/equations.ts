import type {
  BinaryOperator,
  CallExpression,
  CausalEdgeSpec,
  CoefficientExpression,
  CoefficientRole,
  ConstructId,
  ConstructSpec,
  DynamicalModelSpec,
  DynamicsMechanismSpec,
  Expression,
  IndicatorSpec,
  ObservationLawSpec,
  ParameterId,
  ParameterSpec,
} from "@grassmoth/api-types";
import {
  endsOf,
  type MechanismTerm,
  mechanismTerms,
  readingOrder,
} from "@/lib/dag/build-layered-causal-graph";
import { coefficientUses } from "@/lib/model-accessors";
import type { ModelEntities } from "./entities";
import { intervalText, type ParameterLaw, parameterLaw, type StateLaw, stateLaw } from "./laws";
import { observationArguments } from "./observation-law";
import { type EntitySelection, humanize } from "./selection";

/** One piece of a typeset equation. A name carries the entity it refers to. */
export type EquationToken =
  /** An operator, a bracket or other punctuation. */
  | { readonly kind: "symbol"; readonly text: string }
  /** The name of a function, a law, one of its arguments, or a unit. */
  | { readonly kind: "word"; readonly text: string }
  /** A literal, or a fixed coefficient with the role it is authored in. */
  | { readonly kind: "number"; readonly value: number; readonly role: CoefficientRole | null }
  | { readonly kind: "state"; readonly construct: ConstructSpec }
  | { readonly kind: "parameter"; readonly parameter: ParameterSpec }
  | { readonly kind: "unassigned"; readonly role: CoefficientRole }
  | { readonly kind: "superscript"; readonly tokens: readonly EquationToken[] };

/** The tokens a line keeps together: one summand, or one argument of a law. */
type EquationPhrase = readonly EquationToken[];

/** An equation in reading order. A line breaks between phrases before it breaks inside one. */
export type Equation = readonly EquationPhrase[];

/** How tightly an expression's outermost operation binds. */
const LEVEL = { sum: 1, quotient: 2, product: 3, power: 4, atom: 5 } as const;

interface Typeset {
  readonly phrases: Equation;
  /** An operand that binds more loosely than its operator takes brackets. */
  readonly level: number;
  /** It begins with a minus sign, so it takes brackets after another operator. */
  readonly signed: boolean;
}

const FUNCTIONS: Record<CallExpression["function"], string> = {
  exp: "exp",
  sigmoid: "logistic",
  normal_cdf: "Φ",
  ordered_cutpoints: "ordered_cutpoints",
  category_logits: "category_logits",
};

/** An operator keeps its right-hand operand on its line. */
const NBSP = "\u00A0";

const symbol = (text: string): EquationToken => ({ kind: "symbol", text });
const word = (text: string): EquationToken => ({ kind: "word", text });
/** Inside a phrase a line may break before an operator, never after it. */
const infix = (sign: string): EquationToken => symbol(` ${sign}${NBSP}`);
/** A literal writes its signs as mathematics does: a true minus, and no plus on an exponent. */
const signs = (text: string): string => text.replace("-", "−").replace("+", "");

function phrase(tokens: EquationPhrase, level: number, signed = false): Typeset {
  return { phrases: [tokens], level, signed };
}

/** An expression as one run of tokens, as it reads inside a larger one. */
function run(value: Typeset, brackets = false): EquationPhrase {
  const tokens = value.phrases.flatMap((part, index) =>
    index === 0 ? part : [symbol(" "), ...part],
  );
  return brackets ? [symbol("("), ...tokens, symbol(")")] : tokens;
}

function listed(items: readonly EquationPhrase[]): EquationPhrase {
  return items.flatMap((tokens, index) => (index === 0 ? tokens : [symbol(", "), ...tokens]));
}

/** A literal as written: its shortest exact decimal, and its power of ten when it has one. */
export function numberParts(value: number): {
  readonly mantissa: string;
  readonly exponent: string | null;
} {
  const [mantissa = "", exponent] = String(value).split("e");
  return { mantissa: signs(mantissa), exponent: exponent === undefined ? null : signs(exponent) };
}

function numberAtom(value: number, role: CoefficientRole | null): Typeset {
  return phrase([{ kind: "number", value, role }], LEVEL.atom, value < 0);
}

function stateToken(entities: ModelEntities, id: ConstructId): EquationToken {
  const construct = entities.constructById.get(id);
  if (!construct) throw new Error(`Equation references absent construct ${id}`);
  return { kind: "state", construct };
}

/** A parameter reads on the scale its expression uses, so its declared transform is written out. */
function typesetParameter(parameter: ParameterSpec): Typeset {
  const name: EquationToken = { kind: "parameter", parameter };
  switch (parameter.transform.kind) {
    case "dt_persistence_to_ct_decay": {
      const interval = word(intervalText(parameter.transform.interval_days));
      return phrase(
        [symbol("−"), word("log"), symbol("("), name, symbol(")"), infix("/"), interval],
        LEVEL.quotient,
        true,
      );
    }
    case "dt_effect_to_ct_rate": {
      const interval = word(intervalText(parameter.transform.interval_days));
      return phrase([name, infix("/"), interval], LEVEL.quotient);
    }
    case "identity":
    case "initial_state_correlation": {
      return phrase([name], LEVEL.atom);
    }
  }
}

function typesetCoefficient(entities: ModelEntities, operand: CoefficientExpression): Typeset {
  if (operand.value === null) {
    return phrase([{ kind: "unassigned", role: operand.role }], LEVEL.atom);
  }
  if (typeof operand.value === "number") return numberAtom(operand.value, operand.role);
  const parameter = entities.parameterById.get(operand.value);
  if (!parameter) throw new Error(`Equation references absent parameter ${operand.value}`);
  return typesetParameter(parameter);
}

/** A numeric exponent is raised; any other exponent follows a caret. */
function raised(base: Typeset, exponent: Typeset): Typeset {
  const tokens = run(exponent);
  const [only] = tokens;
  const numeric = tokens.length === 1 && only?.kind === "number";
  return phrase(
    [
      ...run(base, base.level < LEVEL.atom || base.signed),
      ...(numeric
        ? [{ kind: "superscript", tokens } as const]
        : [symbol("^"), ...run(exponent, exponent.level < LEVEL.atom || exponent.signed)]),
    ],
    LEVEL.power,
  );
}

function typesetBinary(operator: BinaryOperator, left: Typeset, right: Typeset): Typeset {
  switch (operator) {
    case "add":
    case "subtract": {
      const summand = run(right, right.level <= LEVEL.sum || right.signed);
      return {
        phrases: [
          ...left.phrases,
          [symbol(`${operator === "add" ? "+" : "−"}${NBSP}`), ...summand],
        ],
        level: LEVEL.sum,
        signed: left.signed,
      };
    }
    case "multiply":
    case "divide": {
      // A quotient never borders a product unbracketed, where its extent would be ambiguous.
      const bracketLeft = left.level < LEVEL.product;
      return phrase(
        [
          ...run(left, bracketLeft),
          infix(operator === "divide" ? "/" : "·"),
          ...run(right, right.level < LEVEL.power || right.signed),
        ],
        operator === "divide" ? LEVEL.quotient : LEVEL.product,
        left.signed && !bracketLeft,
      );
    }
    case "power": {
      return raised(left, right);
    }
    case "maximum": {
      return phrase(
        [word("max"), symbol("("), ...listed([run(left), run(right)]), symbol(")")],
        LEVEL.atom,
      );
    }
  }
}

/**
 * Typeset the saved syntax tree without simplifying or evaluating it. Brackets stay wherever
 * reading left to right would regroup the tree, and the product of −1 reads as a negation.
 */
function typeset(entities: ModelEntities, expression: Expression): Typeset {
  switch (expression.kind) {
    case "literal": {
      return numberAtom(expression.value, null);
    }
    case "state": {
      return phrase([stateToken(entities, expression.construct_id)], LEVEL.atom);
    }
    case "coefficient": {
      return typesetCoefficient(entities, expression);
    }
    case "binary": {
      const right = typeset(entities, expression.right);
      if (
        expression.operator === "multiply" &&
        expression.left.kind === "literal" &&
        expression.left.value === -1
      ) {
        const negated = run(right, right.level < LEVEL.product || right.signed);
        return phrase([symbol("−"), ...negated], LEVEL.product, true);
      }
      return typesetBinary(expression.operator, typeset(entities, expression.left), right);
    }
    case "call": {
      const operands = expression.arguments.map((argument) => run(typeset(entities, argument)));
      return phrase(
        [word(FUNCTIONS[expression.function]), symbol("("), ...listed(operands), symbol(")")],
        LEVEL.atom,
      );
    }
  }
}

/** A potential contributes the negative of its gradient in the state it is declared on. */
export function mechanismEquation(
  entities: ModelEntities,
  mechanism: DynamicsMechanismSpec,
  target: ConstructSpec,
  afterPlus = false,
): Equation {
  const expression = typeset(entities, mechanism.expression);
  const contribution =
    mechanism.kind === "drift"
      ? expression
      : phrase(
          [
            symbol("−∂/∂"),
            { kind: "state", construct: target },
            symbol(" ["),
            ...run(expression),
            symbol("]"),
          ],
          LEVEL.product,
          true,
        );
  return afterPlus && contribution.signed ? [run(contribution, true)] : contribution.phrases;
}

/** A law's arguments are its phrases; the first carries the law's name. */
function typesetLaw(
  entities: ModelEntities,
  law: ObservationLawSpec,
  expectedArgument?: string,
): Equation {
  const operands = observationArguments(law);
  return operands.map(([name, value], index) => [
    ...(index === 0 ? [word(law.distribution), symbol("(")] : []),
    word(name),
    symbol(`${NBSP}=${NBSP}`),
    ...(name === expectedArgument ? [word("expected reading")] : run(typeset(entities, value))),
    symbol(index === operands.length - 1 ? ")" : ","),
  ]);
}

/** The response and the conditional law shown by the two measurement nodes. */
export function readingLine(
  entities: ModelEntities,
  indicator: IndicatorSpec,
  split: boolean,
): ReadingLine {
  const law = indicator.likelihood?.law;
  if (!law) return { indicator, expected: null, law: null };
  const reading = (expected: Equation | null): ReadingLine => ({
    indicator,
    expected,
    law: typesetLaw(entities, law),
  });
  if (!split) return reading(null);
  const direct = (name: string, expr: Expression): ReadingLine => ({
    indicator,
    expected: typeset(entities, expr).phrases,
    law: typesetLaw(entities, law, name),
  });
  const operand = (expr: Expression) => typeset(entities, expr);
  switch (law.distribution) {
    case "Delta": {
      return direct("v", law.v);
    }
    case "Normal": {
      return direct("loc", law.loc);
    }
    case "Poisson": {
      return direct("rate", law.rate);
    }
    case "BernoulliProbs": {
      return direct("probs", law.probs);
    }
    case "NegativeBinomial2": {
      return direct("mean", law.mean);
    }
    case "StudentT": {
      return reading([
        ...operand(law.loc).phrases,
        [
          symbol(`if${NBSP}`),
          ...run(operand(law.df)),
          symbol(" > 1"),
          word(", undefined otherwise"),
        ],
      ]);
    }
    case "BernoulliLogits": {
      return reading([[word("logistic"), symbol("("), ...run(operand(law.logits)), symbol(")")]]);
    }
    case "Gamma": {
      return reading(
        typesetBinary("divide", operand(law.concentration), operand(law.rate)).phrases,
      );
    }
    case "Beta": {
      const a = operand(law.concentration1);
      return reading(
        typesetBinary("divide", a, typesetBinary("add", a, operand(law.concentration0))).phrases,
      );
    }
    case "OrderedLogistic":
    case "Categorical": {
      // The response is the expected level index under the displayed observation law.
      return reading([
        [symbol("Σₖ k · P("), word(humanize(indicator.observation.name)), symbol(" = k)")],
      ]);
    }
  }
}

/** A mechanism as one term of the drift of the construct it drives. */
interface DriftTerm {
  readonly term: MechanismTerm;
  readonly equation: Equation;
}

/** An incoming edge that declares no mechanism and that no term reads: only the link is authored. */
interface EdgeLink {
  readonly edge: CausalEdgeSpec;
  readonly cause: ConstructSpec;
}

/** One of a construct's own coefficients: its diffusion or its initial state. */
interface CoefficientLine {
  readonly operand: CoefficientExpression;
  /** The other construct of a diffusion loading or an initial correlation. */
  readonly partner: ConstructSpec | null;
  readonly equation: Equation;
}

/** An indicator with its observation law; null while it has no likelihood. */
interface ReadingLine {
  readonly indicator: IndicatorSpec;
  readonly expected: Equation | null;
  readonly law: Equation | null;
}

interface ParameterLine {
  readonly parameter: ParameterSpec;
  /** The entity whose line names the parameter first; its own law section displays it. */
  readonly owner: EntitySelection;
  readonly law: ParameterLaw;
}

/** One construct's part of the model, in the order its card reads: what drives it, then its readings. */
export interface ConstructEquations {
  readonly construct: ConstructSpec;
  readonly law: StateLaw | null;
  readonly terms: readonly DriftTerm[];
  readonly links: readonly EdgeLink[];
  readonly coefficients: readonly CoefficientLine[];
  readonly readings: readonly ReadingLine[];
  /** Laws of the parameters this construct's lines are the first to name. */
  readonly parameters: readonly ParameterLine[];
}

export function coefficientLine(
  entities: ModelEntities,
  operand: CoefficientExpression,
): CoefficientLine {
  const [other] = operand.construct_ids;
  const partner = other === undefined ? null : entities.constructById.get(other);
  if (partner === undefined) throw new Error(`Coefficient references absent construct ${other}`);
  return { operand, partner, equation: typesetCoefficient(entities, operand).phrases };
}

/** Each line's expressions with the entity that owns them, in reading order. */
function ownedExpressions(construct: ConstructSpec, terms: readonly MechanismTerm[]) {
  const own = { kind: "construct", id: construct.id } as const;
  return [
    ...terms.map((term) => ({ owner: term.owner, expressions: [term.mechanism.expression] })),
    { owner: own, expressions: construct.coefficients },
    ...construct.indicators.map((indicator) => ({
      owner: { kind: "indicator", id: indicator.observation.id } as const,
      expressions: indicator.likelihood
        ? observationArguments(indicator.likelihood.law).map(([, value]) => value)
        : [],
    })),
  ];
}

/**
 * The authored model as text, one block per construct in the graph's reading order. A mechanism
 * reads as a term of the drift of the construct it drives, as its card shows it as a row. Each
 * parameter's law follows the first block that names it.
 */
export function modelEquations(
  model: DynamicalModelSpec,
  entities: ModelEntities,
): readonly ConstructEquations[] {
  const order = readingOrder(entities.constructs, entities.edges, model.outcome);
  const rank = new Map(order.map((construct, index) => [construct.id, index] as const));
  const named = new Set<ParameterId>();
  return order.map((construct) => {
    const terms = mechanismTerms(construct, entities.edges, rank);
    const parameters = ownedExpressions(construct, terms).flatMap(({ owner, expressions }) =>
      coefficientUses(expressions).flatMap(({ parameterId }): ParameterLine[] => {
        const parameter = entities.parameterById.get(parameterId);
        if (!parameter) throw new Error(`Equation references absent parameter ${parameterId}`);
        if (named.has(parameterId)) return [];
        named.add(parameterId);
        return [{ parameter, owner, law: parameterLaw(model, parameter) }];
      }),
    );
    return {
      construct,
      law: stateLaw(model, construct),
      terms: terms.map((term, index) => ({
        term,
        equation: mechanismEquation(entities, term.mechanism, construct, index > 0),
      })),
      links: entities.edges
        .filter((edge) => edge.effect.id === construct.id && endsOf(edge, terms).length === 0)
        .toSorted((a, b) => (rank.get(a.cause.id) ?? 0) - (rank.get(b.cause.id) ?? 0))
        .map((edge) => {
          const cause = entities.constructById.get(edge.cause.id);
          if (!cause) throw new Error(`Edge endpoint ${edge.cause.id} is not a construct.`);
          return { edge, cause };
        }),
      coefficients: construct.coefficients.map((operand) => coefficientLine(entities, operand)),
      readings: construct.indicators.map((indicator) => readingLine(entities, indicator, false)),
      parameters,
    };
  });
}
