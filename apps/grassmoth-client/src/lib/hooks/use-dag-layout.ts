import { useEffect, useMemo, useRef, useState } from "react";
import {
  type DagGraphInput,
  type DagLayoutResult,
  EMPTY_LAYOUT,
  runDagLayout,
} from "@/lib/utils/dag-graph-layout";

interface UseDagLayoutResult extends DagLayoutResult {
  readonly isLayouting: boolean;
}

/**
 * Async ELK layout for the bespoke DAG renderer. Re-layouts whenever the
 * serialized graph changes and ignores results that arrive after a newer input.
 * A failed layout is a bug, so it is rethrown during render for React to surface.
 */
export function useDagLayout(graph: DagGraphInput): UseDagLayoutResult {
  const [result, setResult] = useState<{ data: DagLayoutResult; key: string }>({
    data: EMPTY_LAYOUT,
    key: "",
  });
  const [failure, setFailure] = useState<{ error: unknown } | null>(null);

  const inputKey = useMemo(() => JSON.stringify(graph), [graph]);
  const latestKeyRef = useRef(inputKey);

  useEffect(() => {
    latestKeyRef.current = inputKey;
  });

  useEffect(() => {
    const currentKey = inputKey;
    runDagLayout(graph)
      .then((layout) => {
        if (latestKeyRef.current === currentKey) {
          setResult({ data: layout, key: currentKey });
        }
      })
      .catch((error: unknown) => {
        setFailure({ error });
      });
  }, [inputKey, graph]);

  if (failure !== null) throw failure.error;
  const isLayouting = result.key !== inputKey;
  return { ...result.data, isLayouting };
}
