import type { ConstructSpec, ParameterId } from "@grassmoth/api-types";
import { Fragment, type ReactNode } from "react";
import type { ModelEntities } from "@/lib/model-asset/entities";
import { type Equation, type EquationToken, numberParts } from "@/lib/model-asset/equations";
import type { ParameterLaw, StateLaw } from "@/lib/model-asset/laws";
import { type EntitySelection, humanize } from "@/lib/model-asset/selection";
import { cn } from "@/lib/utils";
import { distributionArgumentText, distributionText } from "@/lib/utils/distribution-format";
import { formatModelDate } from "@/lib/utils/format";

/** What every line of the equations needs: the selection it shows and changes. */
export interface EquationView {
  readonly entities: ModelEntities;
  readonly selection: EntitySelection | null;
  /** The parameters named by the selected entity's own laws. */
  readonly named: ReadonlySet<ParameterId>;
  /** Selects as the graph does: selecting the selected entity clears the selection. */
  readonly toggle: (target: EntitySelection) => void;
}

export function isSelected(selection: EntitySelection | null, target: EntitySelection): boolean {
  return selection?.kind === target.kind && selection.id === target.id;
}

/**
 * A name stays one unit however many words it has. It is inline text, not a box, so a line
 * never breaks between a name and the bracket or sign that belongs to it.
 */
export const NAME = "rounded-[5px] px-1 py-px whitespace-nowrap transition-colors";
export const QUIET = "text-muted-foreground";
/** The selected entity's name, wherever an equation reads it. */
export const PICKED = "bg-primary font-medium text-primary-foreground";
const STATE = "bg-foreground/[0.07] font-medium hover:bg-foreground/[0.14]";

/** A name or caption that selects its entity. A button element would break lines as a box. */
export function Pick({
  view,
  target,
  className,
  children,
}: {
  readonly view: EquationView;
  readonly target: EntitySelection;
  readonly className: string;
  readonly children: ReactNode;
}) {
  return (
    <span
      role="button"
      tabIndex={0}
      aria-pressed={isSelected(view.selection, target)}
      onClick={(event) => {
        event.stopPropagation();
        view.toggle(target);
      }}
      onKeyDown={(event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        event.stopPropagation();
        view.toggle(target);
      }}
      className={cn(
        "cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
    >
      {children}
    </span>
  );
}

/** A construct's name; it selects the construct wherever an equation reads its state. */
export function StateName({
  view,
  construct,
  className,
  children,
}: {
  readonly view: EquationView;
  readonly construct: ConstructSpec;
  readonly className?: string;
  readonly children?: ReactNode;
}) {
  const target = { kind: "construct", id: construct.id } as const;
  return (
    <Pick
      view={view}
      target={target}
      className={cn(NAME, isSelected(view.selection, target) ? PICKED : STATE, className)}
    >
      {children}
      {humanize(construct.name)}
    </Pick>
  );
}

/** The selected entity's own lines carry the selection colour at their edge, as its card does. */
export const SELECTED = "shadow-[inset_2px_0_0_var(--primary)]";

/** A line in the model-wide equations navigates to its owner. */
export function Line({
  view,
  target,
  mark = null,
  className,
  children,
}: {
  readonly view: EquationView;
  readonly target: EntitySelection;
  readonly mark?: "selected" | "named" | null;
  readonly className?: string;
  readonly children: ReactNode;
}) {
  const selected = mark === "selected";
  return (
    <div
      data-selected={selected || undefined}
      onClick={() => {
        if (document.getSelection()?.isCollapsed !== false) view.toggle(target);
      }}
      className={cn(
        "cursor-pointer rounded-[5px] px-1.5 py-px hover:bg-foreground/[0.045]",
        (selected || mark === "named") && "bg-foreground/[0.07]",
        selected && SELECTED,
        className,
      )}
    >
      {children}
    </div>
  );
}

/** A mechanism's or an edge's name above its term; the term's keyboard control. */
export function Caption({
  view,
  target,
  children,
}: {
  readonly view: EquationView;
  readonly target: EntitySelection;
  readonly children: ReactNode;
}) {
  return (
    <Pick
      view={view}
      target={target}
      className="block w-fit rounded-sm pl-1 text-[10px] leading-[1.5] text-muted-foreground hover:text-foreground"
    >
      {children}
    </Pick>
  );
}

/** An operand, mechanism or law the model does not assign yet. */
export function Unassigned({
  title,
  children,
}: {
  readonly title: string;
  readonly children: ReactNode;
}) {
  return (
    <span
      title={title}
      className={cn(NAME, QUIET, "border border-dashed border-muted-foreground/60")}
    >
      {children}
    </span>
  );
}

/** A longer exogenous path is read from its chart; its line gives the extent. */
const PATH_POINTS = 12;
const NBSP = "\u00A0";
export function StateLawText({ law }: { readonly law: StateLaw }) {
  if (law.kind === "law") {
    return (
      <div className="pl-1">
        <span className={QUIET}>~{NBSP}</span>
        <span className="text-foreground/80">{law.law.distribution}</span>
      </div>
    );
  }
  const at = (time: number) =>
    law.timeOrigin === "relative" ? `day ${time}` : formatModelDate(time, law.timeOrigin);
  const first = law.points[0];
  const last = law.points.at(-1);
  return (
    <div className="pl-5 -indent-4">
      <span className={QUIET}>={NBSP}</span>
      {law.points.length > PATH_POINTS && first && last ? (
        <span className="text-foreground/80">
          {law.points.length} values from {at(first.time)} to {at(last.time)}
        </span>
      ) : (
        law.points.map(({ time, value }, index) => (
          <span key={time}>
            {index > 0 && <span className={QUIET}>, </span>}
            <span className="whitespace-nowrap">
              <span className="tabular-nums">{distributionArgumentText(value)}</span>
              <span className={QUIET}> from {at(time)}</span>
            </span>
          </span>
        ))
      )}
    </div>
  );
}

export function ParameterLawText({ law }: { readonly law: ParameterLaw }) {
  switch (law.kind) {
    case "scalar": {
      return <span className="text-foreground/80">{distributionText(law.law)}</span>;
    }
    case "joint": {
      return <span className="text-foreground/80">joint {law.law.distribution}</span>;
    }
    case "unassigned": {
      return <Unassigned title="No law assigned">?</Unassigned>;
    }
  }
}

function Token({
  view,
  token,
}: {
  readonly view: EquationView | null;
  readonly token: EquationToken;
}) {
  switch (token.kind) {
    case "symbol": {
      return <span className={QUIET}>{token.text}</span>;
    }
    case "word": {
      return <span className="whitespace-nowrap text-foreground/80">{token.text}</span>;
    }
    case "number": {
      const { mantissa, exponent } = numberParts(token.value);
      return (
        <span
          title={token.role === null ? undefined : humanize(token.role)}
          className="whitespace-nowrap tabular-nums"
        >
          {mantissa}
          {exponent !== null && (
            <>
              {" × 10"}
              <sup>{exponent}</sup>
            </>
          )}
        </span>
      );
    }
    case "state": {
      return view ? (
        <StateName view={view} construct={token.construct} />
      ) : (
        <span className="font-medium">{humanize(token.construct.name)}</span>
      );
    }
    case "parameter": {
      if (!view) return <span className="font-medium">{humanize(token.parameter.name)}</span>;
      return (
        <span
          className={cn(
            NAME,
            "border",
            view.named.has(token.parameter.id) ? "border-primary" : "border-foreground/20",
          )}
        >
          {humanize(token.parameter.name)}
        </span>
      );
    }
    case "unassigned": {
      return <Unassigned title="Unassigned">? {humanize(token.role)}</Unassigned>;
    }
    case "superscript": {
      return (
        <sup>
          <Tokens view={view} tokens={token.tokens} />
        </sup>
      );
    }
  }
}

function Tokens({
  view,
  tokens,
}: {
  readonly view: EquationView | null;
  readonly tokens: readonly EquationToken[];
}) {
  return tokens.map((token, index) => (
    // eslint-disable-next-line react/no-array-index-key -- tokens are positional in a fixed equation, and a name or sign can repeat
    <Token key={index} view={view} token={token} />
  ));
}

/** Each phrase stays whole while it fits a line, so a long equation breaks between its terms. */
export function Phrases({
  view,
  equation,
}: {
  readonly view: EquationView | null;
  readonly equation: Equation;
}) {
  return equation.map((phrase, index) => (
    // eslint-disable-next-line react/no-array-index-key -- phrases are positional in a fixed equation
    <Fragment key={index}>
      {index > 0 && " "}
      <span className="inline-block max-w-full indent-0 align-top">
        <Tokens view={view} tokens={phrase} />
      </span>
    </Fragment>
  ));
}
