import type { ActionId } from "@grassmoth/api-types";
import {
  ChartNoAxesCombined,
  FileInput,
  GitCompareArrows,
  type LucideIcon,
  Network,
  Play,
  SlidersHorizontal,
  X,
} from "lucide-react";

/** Each action's color and icon, shared by the timeline and the action record. */
export const ACTION_STYLE: Record<ActionId, { color: string; icon: LucideIcon }> = {
  edit_model: { color: "#475569", icon: Network },
  prepare_data: { color: "#0f766e", icon: FileInput },
  fit: { color: "#2563eb", icon: SlidersHorizontal },
  simulate: { color: "#7c3aed", icon: Play },
  data_diff: { color: "#0369a1", icon: ChartNoAxesCombined },
  model_diff: { color: "#b45309", icon: GitCompareArrows },
};

/** Keep the action identifiable when a failure adds a status badge. */
export function ActionMark({
  icon: Icon,
  color,
  failed,
  size = 16,
}: {
  icon: LucideIcon;
  color: string;
  failed?: boolean;
  size?: number;
}) {
  return (
    <span aria-hidden="true" className="relative flex-none">
      <Icon size={size} color={color} />
      {failed && (
        <X
          size={10}
          strokeWidth={3}
          className="absolute -right-1.5 -bottom-1 rounded-full bg-red-50 text-red-600"
        />
      )}
    </span>
  );
}
