import { LLM } from "../llms/base";
import { LLMReranker } from "./llm";

/**
 * Fake LLM that scores a document by looking up the document text inside the
 * prompt. Returns the score as a plain string, mirroring how the OSS LLMs'
 * `generateResponse` returns `response.content` for a plain chat completion.
 *
 * Test document tokens must be distinct and must not be substrings of the
 * prompt boilerplate (e.g. avoid "a"/"b"), or the lookup resolves the wrong doc.
 */
function makeLLM(scoreByDoc: Record<string, string>): LLM {
  return {
    generateResponse: async (
      messages: Array<{ role: string; content: string }>,
    ) => {
      const prompt = messages.map((m) => m.content).join("\n");
      const doc = Object.keys(scoreByDoc).find((d) => prompt.includes(d));
      return doc ? scoreByDoc[doc] : "no number here";
    },
    generateChat: async () => ({ content: "", role: "assistant" }),
  };
}

describe("LLMReranker", () => {
  it("sorts documents by descending relevance score", async () => {
    const llm = makeLLM({ cats: "0.2", dogs: "0.9", fish: "0.5" });
    const reranker = new LLMReranker({}, llm);

    const results = await reranker.rerank("pets", ["cats", "dogs", "fish"]);

    expect(results.map((r) => r.index)).toEqual([1, 2, 0]);
    expect(results.map((r) => r.relevanceScore)).toEqual([0.9, 0.5, 0.2]);
  });

  it("clamps scores to the [0, 1] range", async () => {
    const llm = makeLLM({ zebra: "1.5", walrus: "-0.3" });
    const reranker = new LLMReranker({}, llm);

    const results = await reranker.rerank("q", ["zebra", "walrus"]);

    const byIndex = new Map(results.map((r) => [r.index, r.relevanceScore]));
    expect(byIndex.get(0)).toBe(1); // "zebra" 1.5 -> clamped to 1
    expect(byIndex.get(1)).toBe(0); // "walrus" -0.3 -> clamped to 0
  });

  it("truncates results to topN", async () => {
    const llm = makeLLM({ alpha: "0.1", bravo: "0.8", charlie: "0.5" });
    const reranker = new LLMReranker({}, llm);

    const results = await reranker.rerank(
      "q",
      ["alpha", "bravo", "charlie"],
      2,
    );

    expect(results).toHaveLength(2);
    expect(results.map((r) => r.index)).toEqual([1, 2]); // bravo(0.8), charlie(0.5)
  });

  it("falls back to score 0 when the LLM output has no number", async () => {
    const llm = makeLLM({ junk: "I cannot rate this" });
    const reranker = new LLMReranker({}, llm);

    const results = await reranker.rerank("q", ["junk"]);

    expect(results[0].relevanceScore).toBe(0);
  });

  it("throws when no LLM is configured or provided", () => {
    expect(() => new LLMReranker({})).toThrow();
  });
});
