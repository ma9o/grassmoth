import type { DynamicalModelSpec, ParameterId } from "@grassmoth/api-types";
import { type ReactNode, useEffect, useMemo, useRef } from "react";
import { type ModelEntities, resolveEntity } from "@/lib/model-asset/entities";
import { type ConstructEquations, modelEquations } from "@/lib/model-asset/equations";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { type EntitySelection, humanize, mechanismName } from "@/lib/model-asset/selection";
import { glyphKey } from "@/lib/model-asset/spec-plots";
import { cn } from "@/lib/utils";
import {
  Caption,
  type EquationView,
  isSelected,
  Line,
  NAME,
  ParameterLawText,
  Phrases,
  PICKED,
  Pick,
  QUIET,
  SELECTED,
  StateLawText,
  StateName,
  Unassigned,
} from "./equation-text";
import { Section } from "./scope-primitives";

/** A relation keeps its right-hand side on its line. */
const NBSP = "\u00A0";
function selectedEntity(entities: ModelEntities, selection: EntitySelection) {
  switch (selection.kind) {
    case "construct": {
      return entities.constructById.get(selection.id);
    }
    case "edge": {
      return entities.edgeById.get(selection.id);
    }
    case "indicator": {
      return entities.indicatorById.get(selection.id);
    }
  }
}

function namedParameters(
  entities: ModelEntities,
  selection: EntitySelection | null,
): ReadonlySet<ParameterId> {
  const entity = selection ? selectedEntity(entities, selection) : undefined;
  return new Set(entity ? ownLawUses(entity).map((use) => use.parameterId) : []);
}

/** One term of a drift; its wrapped lines hang under its first operand. */
function Term({ lead, children }: { readonly lead: string; readonly children: ReactNode }) {
  return (
    <div className="pl-[52px] -indent-[52px]">
      <span className={cn(QUIET, "mr-1.5 inline-block w-[46px] text-right indent-0 align-top")}>
        {lead}
      </span>
      {children}
    </div>
  );
}

function Tag({ children }: { readonly children: ReactNode }) {
  return <span className="text-[10px] whitespace-nowrap text-muted-foreground">{children}</span>;
}

/** What drives the construct: its mechanisms as the terms of its drift, then its bare links. */
function Drift({
  view,
  block,
}: {
  readonly view: EquationView;
  readonly block: ConstructEquations;
}) {
  const lead = block.construct.temporal_status === "time_varying" ? "drift =" : "=";
  return (
    <>
      {block.terms.map(({ term, equation }, index) => {
        const edge = term.owner.kind === "edge" ? resolveEntity(view.entities, term.owner) : null;
        const selected = view.selection?.glyph
          ? view.selection.glyph === term.mechanism.id
          : edge && isSelected(view.selection, term.owner);
        return (
          <Line
            key={term.mechanism.id}
            view={view}
            target={term.owner}
            mark={selected ? "selected" : null}
          >
            <Caption view={view} target={term.owner}>
              {mechanismName(term.mechanism.id)}
              {edge && ` · ${edge.label}`}
            </Caption>
            <Term lead={index === 0 ? lead : "+"}>
              <Phrases view={view} equation={equation} />
            </Term>
          </Line>
        );
      })}
      {block.links.map(({ edge, cause }, index) => {
        const target = { kind: "edge", id: edge.id } as const;
        return (
          <Line
            key={edge.id}
            view={view}
            target={target}
            mark={isSelected(view.selection, target) ? "selected" : null}
          >
            <Caption view={view} target={target}>
              {resolveEntity(view.entities, target)?.label}
            </Caption>
            <Term lead={block.terms.length + index === 0 ? lead : "+"}>
              <Unassigned title="No mechanism declared">?</Unassigned>
              <span className={QUIET}>(</span>
              <StateName view={view} construct={cause} />
              <span className={QUIET}>)</span>
            </Term>
          </Line>
        );
      })}
    </>
  );
}

/** The construct's readings, then the laws of the parameters its lines are the first to name. */
function Laws({
  view,
  block,
}: {
  readonly view: EquationView;
  readonly block: ConstructEquations;
}) {
  return (
    <>
      {block.readings.map(({ indicator, law }) => {
        const target = { kind: "indicator", id: indicator.observation.id } as const;
        const picked = isSelected(view.selection, target);
        return (
          <Line key={target.id} view={view} target={target} mark={picked ? "selected" : null}>
            <div className="pl-4 -indent-4">
              <Pick
                view={view}
                target={target}
                className={cn(NAME, "font-medium hover:bg-foreground/[0.1]", picked && PICKED)}
              >
                {humanize(indicator.observation.name)}
              </Pick>
              <span className={QUIET}> ~{NBSP}</span>
              {law ? (
                <Phrases view={view} equation={law} />
              ) : (
                <Unassigned title="No likelihood assigned">?</Unassigned>
              )}
              {indicator.likelihood?.standardized && <Tag> · standardized</Tag>}
            </div>
          </Line>
        );
      })}
      {block.parameters.length > 0 && (
        <div className="px-2.5 pt-0.5 text-[10px] text-muted-foreground">where</div>
      )}
      {block.parameters.map(({ parameter, owner, law }) => (
        <Line
          key={parameter.id}
          view={view}
          target={owner}
          mark={view.named.has(parameter.id) ? "named" : null}
        >
          <div className="pl-4 -indent-4">
            <Pick
              view={view}
              target={owner}
              className={cn(NAME, "border border-foreground/20", "hover:bg-foreground/[0.07]")}
            >
              {humanize(parameter.name)}
            </Pick>
            <span className={QUIET}> ~{NBSP}</span>
            <ParameterLawText law={law} />
          </div>
        </Line>
      ))}
    </>
  );
}

/** One construct as its card reads: what drives it, its own coefficients, then its readings. */
function ConstructBlock({
  view,
  block,
  outcome,
}: {
  readonly view: EquationView;
  readonly block: ConstructEquations;
  readonly outcome: boolean;
}) {
  const { construct } = block;
  const own = { kind: "construct", id: construct.id } as const;
  const selected = !view.selection?.glyph && isSelected(view.selection, own);
  const student = construct.innovation_family === "student_t";
  return (
    <div
      data-construct-id={construct.id}
      data-selected={selected || undefined}
      className={cn(
        "flex break-inside-avoid flex-col gap-px border-b border-border/70 py-1.5 last:border-b-0",
        selected && SELECTED,
      )}
    >
      <Line view={view} target={own} className="flex flex-wrap items-baseline gap-x-1.5">
        <StateName view={view} construct={construct} className="text-xs font-semibold">
          {outcome && "★ "}
        </StateName>
        {construct.temporal_status === "time_varying" ? (
          <span role="img" aria-label="time varying" title="time varying" className={QUIET}>
            ↻
          </span>
        ) : (
          <Tag>{humanize(construct.temporal_status)}</Tag>
        )}
        {construct.role === "exogenous" && <Tag>{construct.role}</Tag>}
      </Line>
      {block.law && (
        <Line view={view} target={own}>
          <StateLawText law={block.law} />
        </Line>
      )}
      <Drift view={view} block={block} />
      {(block.coefficients.length > 0 || student) && (
        <Line view={view} target={own} className="flex flex-wrap gap-x-3 pl-2.5">
          {block.coefficients.map(({ operand, partner, equation }) => (
            <span key={`${operand.role}:${partner?.id ?? ""}`}>
              <span className={QUIET}>
                {humanize(operand.role)}
                {partner && " with "}
              </span>
              {partner && <StateName view={view} construct={partner} />}
              <span className={QUIET}> ={NBSP}</span>
              <Phrases view={view} equation={equation} />
            </span>
          ))}
          {student && (
            <span className={QUIET}>
              innovation family ={NBSP}
              <span className="text-foreground/80">{humanize(construct.innovation_family)}</span>
            </span>
          )}
        </Line>
      )}
      <Laws view={view} block={block} />
    </div>
  );
}

/** The navigable model-wide equations shown alongside analysis actions. */
export function AuthoredEquations({
  model,
  context,
  selection,
}: {
  readonly model: DynamicalModelSpec;
  readonly context: ScopeContext;
  readonly selection: EntitySelection | null;
}) {
  const { entities, select } = context;
  const blocks = useMemo(() => modelEquations(model, entities), [model, entities]);
  const named = useMemo(() => namedParameters(entities, selection), [entities, selection]);
  const root = useRef<HTMLDivElement>(null);
  const selected = selection ? glyphKey(selection) : null;
  useEffect(() => {
    if (selected === null) return;
    root.current?.querySelector("[data-selected]")?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  const view: EquationView = {
    entities,
    selection,
    named,
    toggle: (target) => select(selected === glyphKey(target) ? null : target),
  };
  return (
    <Section title="Authored equations">
      <div
        ref={root}
        className="columns-[420px] gap-x-6 leading-[1.8] [column-rule:1px_solid_var(--border)]"
      >
        {blocks.map((block) => (
          <ConstructBlock
            key={block.construct.id}
            view={view}
            block={block}
            outcome={block.construct.id === model.outcome}
          />
        ))}
      </div>
    </Section>
  );
}
