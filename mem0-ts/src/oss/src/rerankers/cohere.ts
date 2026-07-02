import { CohereClientV2 } from "cohere-ai";
import { RerankerConfig } from "../types";
import { Reranker, RerankResult } from "./base";

const DEFAULT_MODEL = "rerank-v3.5";

/**
 * Reranker backed by Cohere's `/v2/rerank` endpoint. Cohere returns relevance
 * scores already normalized to `[0, 1]`, ordered most-relevant first, so results
 * pass straight through.
 */
export class CohereReranker implements Reranker {
  private client: CohereClientV2;
  private model: string;

  constructor(config: RerankerConfig) {
    this.client = new CohereClientV2({ token: config.apiKey });
    this.model = config.model || DEFAULT_MODEL;
  }

  async rerank(
    query: string,
    documents: string[],
    topN?: number,
  ): Promise<RerankResult[]> {
    if (documents.length === 0) return [];

    const response = await this.client.rerank({
      model: this.model,
      query,
      documents,
      topN,
    });

    return response.results.map((result) => ({
      index: result.index,
      relevanceScore: result.relevanceScore,
    }));
  }
}
