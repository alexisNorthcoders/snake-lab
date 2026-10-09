import { type Brain, INPUT_LABELS, OUTPUT_LABELS } from "snake-colyseus/bots";

/** A network as the page draws it: one column of labelled nodes a layer, and an edge for every weight. */
export interface NetworkLayout {
  /** From the input to the output. Hidden nodes are numbered from 1. */
  columns: { labels: string[] }[];
  /** `layer` is the column the edge leaves; `from` and `to` are node places in that column and the next. */
  edges: { layer: number; from: number; to: number; weight: number }[];
}

/** `brain`'s layout: the encoder v1's labels in, `left`, `straight`, `right` out, a node a hidden unit between. */
export function networkLayout(brain: Brain): NetworkLayout {
  const last = brain.sizes.length - 1;
  const columns = brain.sizes.map((size, l) => ({
    labels: l === 0 ? [...INPUT_LABELS] : l === last ? [...OUTPUT_LABELS] : Array.from({ length: size }, (_, i) => String(i + 1))
  }));
  const edges = brain.layers.flatMap(({ weights }, layer) =>
    weights.flatMap((row, to) => row.map((weight, from) => ({ layer, from, to, weight }))));
  return { columns, edges };
}
