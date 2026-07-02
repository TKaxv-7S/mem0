jest.mock("zeroentropy", () => ({ ZeroEntropy: jest.fn() }));
jest.mock("@huggingface/transformers", () => ({
  AutoModelForSequenceClassification: { from_pretrained: jest.fn() },
  AutoTokenizer: { from_pretrained: jest.fn() },
}));

import { RerankerFactory } from "./factory";
import { CohereReranker } from "../rerankers/cohere";
import { LLMReranker } from "../rerankers/llm";
import { ZeroEntropyReranker } from "../rerankers/zeroentropy";
import { CrossEncoderReranker } from "../rerankers/cross_encoder";
import { LLM } from "../llms/base";

const fakeLLM: LLM = {
  generateResponse: async () => "0.5",
  generateChat: async () => ({ content: "", role: "assistant" }),
};

describe("RerankerFactory", () => {
  it("creates a CohereReranker for provider 'cohere'", () => {
    const reranker = RerankerFactory.create("cohere", { apiKey: "key" });
    expect(reranker).toBeInstanceOf(CohereReranker);
  });

  it("matches the provider name case-insensitively", () => {
    const reranker = RerankerFactory.create("Cohere", { apiKey: "key" });
    expect(reranker).toBeInstanceOf(CohereReranker);
  });

  it("creates a ZeroEntropyReranker for provider 'zeroentropy'", () => {
    const reranker = RerankerFactory.create("zeroentropy", { apiKey: "key" });
    expect(reranker).toBeInstanceOf(ZeroEntropyReranker);
  });

  it("creates a CrossEncoderReranker for provider 'sentence_transformer'", () => {
    const reranker = RerankerFactory.create("sentence_transformer", {});
    expect(reranker).toBeInstanceOf(CrossEncoderReranker);
  });

  it("creates a CrossEncoderReranker for provider 'huggingface'", () => {
    const reranker = RerankerFactory.create("huggingface", {});
    expect(reranker).toBeInstanceOf(CrossEncoderReranker);
  });

  it("creates an LLMReranker using the default LLM when config.llm is omitted", () => {
    const reranker = RerankerFactory.create("llm", {}, fakeLLM);
    expect(reranker).toBeInstanceOf(LLMReranker);
  });

  it("creates an LLMReranker that builds its own LLM from config.llm", () => {
    const reranker = RerankerFactory.create("llm", {
      llm: { provider: "openai", config: { apiKey: "x" } },
    });
    expect(reranker).toBeInstanceOf(LLMReranker);
  });

  it("throws for the 'llm' provider when no LLM is available", () => {
    expect(() => RerankerFactory.create("llm", {})).toThrow();
  });

  it("throws for an unsupported provider", () => {
    expect(() => RerankerFactory.create("banana", {})).toThrow(
      /unsupported reranker provider/i,
    );
  });
});
