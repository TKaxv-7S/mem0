export interface RerankResult {
  /** Index into the input `documents` array. */
  index: number;
  /** Relevance of the document to the query, 0..1, higher = more relevant. */
  relevanceScore: number;
}

export interface Reranker {
  /**
   * Rank `documents` by relevance to `query`.
   *
   * Returns results sorted by descending relevance. When `topN` is given, at
   * most that many results are returned. Each result's `index` points back into
   * the input `documents` array so callers can recover the original item.
   */
  rerank(
    query: string,
    documents: string[],
    topN?: number,
  ): Promise<RerankResult[]>;
}
