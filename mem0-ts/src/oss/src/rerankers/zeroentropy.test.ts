const mockRerank = jest.fn();

jest.mock("zeroentropy", () => ({
  ZeroEntropy: jest.fn().mockImplementation(() => ({
    models: { rerank: mockRerank },
  })),
}));

import { ZeroEntropy } from "zeroentropy";
import { ZeroEntropyReranker } from "./zeroentropy";

describe("ZeroEntropyReranker", () => {
  beforeEach(() => {
    mockRerank.mockReset();
    (ZeroEntropy as unknown as jest.Mock).mockClear();
  });

  it("sends the query, documents, topN, and default model to ZeroEntropy", async () => {
    mockRerank.mockResolvedValue({ results: [] });
    const reranker = new ZeroEntropyReranker({ apiKey: "key" });

    await reranker.rerank("capital of US?", ["a", "b", "c"], 2);

    expect(mockRerank).toHaveBeenCalledWith({
      model: "zerank-1",
      query: "capital of US?",
      documents: ["a", "b", "c"],
      top_n: 2,
    });
  });

  it("maps ZeroEntropy's results (relevance_score) to {index, relevanceScore}", async () => {
    mockRerank.mockResolvedValue({
      results: [
        { index: 2, relevance_score: 0.9 },
        { index: 0, relevance_score: 0.31 },
      ],
    });
    const reranker = new ZeroEntropyReranker({ apiKey: "key" });

    const results = await reranker.rerank("q", ["x", "y", "z"]);

    expect(results).toEqual([
      { index: 2, relevanceScore: 0.9 },
      { index: 0, relevanceScore: 0.31 },
    ]);
  });

  it("uses a custom model when provided", async () => {
    mockRerank.mockResolvedValue({ results: [] });
    const reranker = new ZeroEntropyReranker({
      apiKey: "key",
      model: "zerank-1-small",
    });

    await reranker.rerank("q", ["a"]);

    expect(mockRerank).toHaveBeenCalledWith(
      expect.objectContaining({ model: "zerank-1-small" }),
    );
  });

  it("returns an empty array without calling ZeroEntropy when there are no documents", async () => {
    const reranker = new ZeroEntropyReranker({ apiKey: "key" });

    const results = await reranker.rerank("q", []);

    expect(results).toEqual([]);
    expect(mockRerank).not.toHaveBeenCalled();
  });
});
