const mockRerank = jest.fn();

jest.mock("cohere-ai", () => ({
  CohereClientV2: jest.fn().mockImplementation(() => ({
    rerank: mockRerank,
  })),
}));

import { CohereClientV2 } from "cohere-ai";
import { CohereReranker } from "./cohere";

describe("CohereReranker", () => {
  beforeEach(() => {
    mockRerank.mockReset();
    (CohereClientV2 as jest.Mock).mockClear();
  });

  it("sends the query, documents, topN, and default model to Cohere", async () => {
    mockRerank.mockResolvedValue({ results: [] });
    const reranker = new CohereReranker({ apiKey: "key" });

    await reranker.rerank("capital of US?", ["a", "b", "c"], 2);

    expect(mockRerank).toHaveBeenCalledWith({
      model: "rerank-v3.5",
      query: "capital of US?",
      documents: ["a", "b", "c"],
      topN: 2,
    });
  });

  it("returns Cohere's ranked results as {index, relevanceScore}", async () => {
    mockRerank.mockResolvedValue({
      results: [
        { index: 2, relevanceScore: 0.9 },
        { index: 0, relevanceScore: 0.31 },
      ],
    });
    const reranker = new CohereReranker({ apiKey: "key" });

    const results = await reranker.rerank("q", ["x", "y", "z"]);

    expect(results).toEqual([
      { index: 2, relevanceScore: 0.9 },
      { index: 0, relevanceScore: 0.31 },
    ]);
  });

  it("uses a custom model when provided", async () => {
    mockRerank.mockResolvedValue({ results: [] });
    const reranker = new CohereReranker({
      apiKey: "key",
      model: "rerank-v4.0-pro",
    });

    await reranker.rerank("q", ["a"]);

    expect(mockRerank).toHaveBeenCalledWith(
      expect.objectContaining({ model: "rerank-v4.0-pro" }),
    );
  });

  it("returns an empty array without calling Cohere when there are no documents", async () => {
    const reranker = new CohereReranker({ apiKey: "key" });

    const results = await reranker.rerank("q", []);

    expect(results).toEqual([]);
    expect(mockRerank).not.toHaveBeenCalled();
  });
});
