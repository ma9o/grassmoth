import { HelpCircle } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export function StatTooltip({ explanation }: { readonly explanation: string }) {
  return (
    <Tooltip>
      <TooltipTrigger render={<span />} className="inline-flex cursor-help">
        <HelpCircle
          aria-hidden
          className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground transition-colors"
        />
      </TooltipTrigger>
      <TooltipContent className="max-w-xl items-start whitespace-normal">
        <span className="text-xs leading-relaxed break-words">{explanation}</span>
      </TooltipContent>
    </Tooltip>
  );
}
