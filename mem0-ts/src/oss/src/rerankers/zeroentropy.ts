import { ZeroEntropy } from "zeroentropy";
import { RerankerConfig } from "../types";
import { Reranker, RerankResult } from "./base";

const DEFAULT_MODEL = "zerank-1";

/**
 * Reranker backed by ZeroEntropy's `/models/rerank` endpoint. ZeroEntropy
 * returns relevance scores already normalized to `[0, 1]`, ordered
 * most-relevant first, so results pass straight through.
 *
 * The API key is read from `config.apiKey`, falling back to the SDK-native
 * `ZEROENTROPY_API_KEY` or the Python-SDK `ZERO_ENTROPY_API_KEY` env var.
 */
export class ZeroEntropyReranker implements Reranker {
  private client: ZeroEntropy;
  private model: string;

  constructor(config: RerankerConfig) {
    const apiKey =
      config.apiKey ??
      process.env.ZEROENTROPY_API_KEY ??
      process.env.ZERO_ENTROPY_API_KEY;
    this.client = new ZeroEntropy({ apiKey });
    this.model = config.model || DEFAULT_MODEL;
  }

  async rerank(
    query: string,
    documents: string[],
    topN?: number,
  ): Promise<RerankResult[]> {
    if (documents.length === 0) return [];

    const response = await this.client.models.rerank({
      model: this.model,
      query,
      documents,
      top_n: topN,
    });

    return response.results.map((result) => ({
      index: result.index,
      relevanceScore: result.relevance_score,
    }));
  }
}
