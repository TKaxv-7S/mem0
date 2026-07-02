import { RerankerConfig } from "../types";
import { LLM, LLMResponse } from "../llms/base";
import { Reranker, RerankResult } from "./base";

const SCORE_PATTERN = /-?\d+(\.\d+)?/;

/**
 * Reranker that scores each document's relevance to the query with an LLM.
 *
 * The LLM is injected (resolved by `RerankerFactory` from the reranker's own
 * `config.llm`, or the Memory's main `llm`), so this class never builds one
 * itself — that keeps it free of a factory import cycle.
 */
export class LLMReranker implements Reranker {
  private llm: LLM;

  constructor(_config: RerankerConfig, llm?: LLM) {
    if (!llm) {
      throw new Error(
        "LLMReranker requires an LLM: provide one via the reranker `config.llm` or configure a Memory `llm`.",
      );
    }
    this.llm = llm;
  }

  async rerank(
    query: string,
    documents: string[],
    topN?: number,
  ): Promise<RerankResult[]> {
    const scored = await Promise.all(
      documents.map(async (document, index) => ({
        index,
        relevanceScore: await this.score(query, document),
      })),
    );

    scored.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return topN != null ? scored.slice(0, topN) : scored;
  }

  private async score(query: string, document: string): Promise<number> {
    const messages = [
      {
        role: "user",
        content:
          "On a scale from 0.0 to 1.0, how relevant is the following document " +
          "to the query? Respond with only the number.\n\n" +
          `Query: ${query}\n\nDocument: ${document}`,
      },
    ];

    const response = await this.llm.generateResponse(messages);
    const text =
      typeof response === "string"
        ? response
        : ((response as LLMResponse)?.content ?? "");

    const match = text.match(SCORE_PATTERN);
    if (!match) return 0;

    const value = parseFloat(match[0]);
    if (Number.isNaN(value)) return 0;
    return Math.min(1, Math.max(0, value));
  }
}
