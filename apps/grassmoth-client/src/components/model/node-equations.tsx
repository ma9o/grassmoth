import type {
  ConstructSpec,
  DynamicalModelSpec,
  DynamicsMechanismSpec,
  IndicatorSpec,
  ParameterId,
  ParameterSpec,
} from "@grassmoth/api-types";
import type { ModelEntities } from "@/lib/model-asset/entities";
import {
  coefficientLine,
  type Equation,
  type EquationToken,
  mechanismEquation,
  readingLine,
} from "@/lib/model-asset/equations";
import { parameterLaw, stateLaw } from "@/lib/model-asset/laws";
import { type EntitySelection, humanize, mechanismName } from "@/lib/model-asset/selection";
import { ParameterLawText, Phrases, StateLawText } from "./equation-text";
import { Hint, Section } from "./scope-primitives";

interface Formula {
  readonly key: string;
  readonly label: string;
  readonly operator: "=" | "~";
  readonly equation: Equation | null;
  readonly standardized?: boolean;
}

function measurement(
  entities: ModelEntities,
  indicator: IndicatorSpec,
  split: boolean,
  response = false,
): Formula {
  const { expected, law } = readingLine(entities, indicator, split);
  const name = humanize(indicator.observation.name);
  return {
    key: indicator.observation.id,
    label: response ? `Expected reading for ${name}` : name,
    operator: response ? "=" : "~",
    equation: response ? expected : law,
    standardized: indicator.likelihood?.standardized === true,
  };
}

function contribution(
  entities: ModelEntities,
  mechanism: DynamicsMechanismSpec,
  effect: ConstructSpec,
): Formula {
  return {
    key: mechanism.id,
    label: `${mechanismName(mechanism.id)} (${effect.temporal_status === "time_varying" ? "drift term" : "term"} for ${humanize(effect.name)})`,
    operator: "=",
    equation: mechanismEquation(entities, mechanism, effect),
  };
}

/** In the compact DAG mechanisms and readings are components of their construct. */
function localFormulas(
  entities: ModelEntities,
  selection: EntitySelection,
  separateNodes: boolean,
): Formula[] {
  if (selection.kind === "indicator") {
    const indicator = entities.indicatorById.get(selection.id);
    if (!indicator) return [];
    const split =
      separateNodes && entities.indicatorOwnerById.get(selection.id)?.role === "endogenous";
    return [measurement(entities, indicator, split, selection.glyph === "response")];
  }
  const edge = selection.kind === "edge" ? entities.edgeById.get(selection.id) : undefined;
  const effect =
    selection.kind === "construct"
      ? entities.constructById.get(selection.id)
      : edge && entities.constructById.get(edge.effect.id);
  if (!effect) return [];
  if (selection.glyph || edge) {
    return (edge?.mechanisms ?? effect.dynamics)
      .filter((m) => !selection.glyph || m.id === selection.glyph)
      .map((m) => contribution(entities, m, effect));
  }
  const coefficients = effect.coefficients.map((operand): Formula => {
    const { partner, equation } = coefficientLine(entities, operand);
    return {
      key: `${operand.role}/${partner?.id ?? ""}`,
      label: humanize(operand.role) + (partner ? ` with ${humanize(partner.name)}` : ""),
      operator: "=",
      equation,
    };
  });
  if (separateNodes && effect.role === "endogenous") return coefficients;
  const terms = [
    ...effect.dynamics,
    ...entities.edges
      .filter((incoming) => incoming.effect.id === effect.id)
      .flatMap((incoming) => incoming.mechanisms),
  ];
  return [
    ...coefficients,
    ...terms.map((m) => contribution(entities, m, effect)),
    ...effect.indicators.map((indicator) => measurement(entities, indicator, false)),
  ];
}

/** Include each parameter law where its symbol occurs, including laws shared by several nodes. */
function formulaParameters(formulas: readonly Formula[]) {
  const parameters = new Map<ParameterId, ParameterSpec>();
  const collect = (tokens: readonly EquationToken[]) => {
    for (const token of tokens) {
      if (token.kind === "parameter") parameters.set(token.parameter.id, token.parameter);
      if (token.kind === "superscript") collect(token.tokens);
    }
  };
  for (const formula of formulas) collect(formula.equation?.flat() ?? []);
  return [...parameters.values()];
}

export function NodeEquations({
  model,
  entities,
  selection,
  separateNodes,
}: {
  readonly model: DynamicalModelSpec;
  readonly entities: ModelEntities;
  readonly selection: EntitySelection;
  readonly separateNodes: boolean;
}) {
  const state =
    selection.kind === "construct" && !selection.glyph
      ? entities.constructById.get(selection.id)
      : undefined;
  const law = state && stateLaw(model, state);
  const formulas = localFormulas(entities, selection, separateNodes);
  const parameters = formulaParameters(formulas);
  return (
    <Section title="Equations" wide>
      {law && (
        <div>
          <span className="font-medium">{humanize(state.name)}</span>
          <StateLawText law={law} />
        </div>
      )}
      {formulas.map(({ key, label, operator, equation, standardized }) => (
        <p key={key} className="leading-relaxed">
          <span className="font-medium">{label}</span>
          <span className="text-muted-foreground"> {operator}&nbsp;</span>
          {equation ? (
            <Phrases view={null} equation={equation} />
          ) : (
            <span className="text-muted-foreground">? (no likelihood assigned)</span>
          )}
          {standardized && <span className="text-muted-foreground"> · standardized</span>}
        </p>
      ))}
      {state?.innovation_family === "student_t" && <p>Innovation family = Student t</p>}
      {!law && formulas.length === 0 && state?.innovation_family !== "student_t" && (
        <Hint>No equation authored for this node.</Hint>
      )}
      {parameters.length > 0 && (
        <div className="space-y-1 border-t border-border/60 pt-2">
          {parameters.map((parameter) => (
            <p key={parameter.id}>
              {humanize(parameter.name)} <span className="text-muted-foreground">~&nbsp;</span>
              <ParameterLawText law={parameterLaw(model, parameter)} />
            </p>
          ))}
        </div>
      )}
    </Section>
  );
}
