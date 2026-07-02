const mockModelFromPretrained = jest.fn();
const mockTokenizerFromPretrained = jest.fn();

jest.mock("@huggingface/transformers", () => ({
  AutoModelForSequenceClassification: {
    from_pretrained: mockModelFromPretrained,
  },
  AutoTokenizer: { from_pretrained: mockTokenizerFromPretrained },
}));

import { CrossEncoderReranker } from "./cross_encoder";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Wire the mocked tokenizer + model so the model returns `logits` for a call. */
function setupModel(logits: number[][]) {
  const tokenizer = jest.fn().mockReturnValue({ input_ids: [] });
  mockTokenizerFromPretrained.mockResolvedValue(tokenizer);
  const model = jest
    .fn()
    .mockResolvedValue({ logits: { tolist: () => logits } });
  mockModelFromPretrained.mockResolvedValue(model);
  return { tokenizer, model };
}

describe("CrossEncoderReranker", () => {
  beforeEach(() => {
    mockModelFromPretrained.mockReset();
    mockTokenizerFromPretrained.mockReset();
  });

  it("scores each document and returns them sorted by relevance, sigmoid-normalized to [0,1]", async () => {
    setupModel([[0.0], [2.0], [-1.0]]);
    const reranker = new CrossEncoderReranker({}, "default-model");

    const results = await reranker.rerank("q", ["a", "b", "c"]);

    // sigmoid: b(2.0)=0.88 > a(0.0)=0.5 > c(-1.0)=0.27
    expect(results.map((r) => r.index)).toEqual([1, 0, 2]);
    expect(results[0].relevanceScore).toBeCloseTo(sigmoid(2.0), 5);
    expect(results[1].relevanceScore).toBeCloseTo(sigmoid(0.0), 5);
    expect(results[2].relevanceScore).toBeCloseTo(sigmoid(-1.0), 5);
  });

  it("pairs the query with each document via text_pair when tokenizing", async () => {
    const { tokenizer } = setupModel([[0.1], [0.2]]);
    const reranker = new CrossEncoderReranker(
      { maxLength: 128 },
      "default-model",
    );

    await reranker.rerank("what is x", ["doc one", "doc two"]);

    expect(tokenizer).toHaveBeenCalledWith(
      ["what is x", "what is x"],
      expect.objectContaining({
        text_pair: ["doc one", "doc two"],
        padding: true,
        truncation: true,
        max_length: 128,
      }),
    );
  });

  it("applies the topN limit", async () => {
    setupModel([[0.0], [2.0], [-1.0]]);
    const reranker = new CrossEncoderReranker({}, "default-model");

    const results = await reranker.rerank("q", ["a", "b", "c"], 2);

    expect(results).toHaveLength(2);
    expect(results.map((r) => r.index)).toEqual([1, 0]);
  });

  it("returns [] without loading the model when there are no documents", async () => {
    const reranker = new CrossEncoderReranker({}, "default-model");

    const results = await reranker.rerank("q", []);

    expect(results).toEqual([]);
    expect(mockModelFromPretrained).not.toHaveBeenCalled();
    expect(mockTokenizerFromPretrained).not.toHaveBeenCalled();
  });

  it("returns raw logits as scores when normalize is false", async () => {
    setupModel([[2.0], [0.0]]);
    const reranker = new CrossEncoderReranker(
      { normalize: false },
      "default-model",
    );

    const results = await reranker.rerank("q", ["a", "b"]);

    expect(results.map((r) => r.index)).toEqual([0, 1]);
    expect(results[0].relevanceScore).toBe(2.0);
    expect(results[1].relevanceScore).toBe(0.0);
  });

  it("loads the model and tokenizer only once across multiple rerank calls", async () => {
    setupModel([[0.5]]);
    const reranker = new CrossEncoderReranker({}, "default-model");

    await reranker.rerank("q", ["a"]);
    await reranker.rerank("q2", ["b"]);

    expect(mockModelFromPretrained).toHaveBeenCalledTimes(1);
    expect(mockTokenizerFromPretrained).toHaveBeenCalledTimes(1);
  });

  it("loads the default model, or the configured model when provided", async () => {
    setupModel([[0.5]]);

    await new CrossEncoderReranker({}, "the-default").rerank("q", ["a"]);
    expect(mockModelFromPretrained).toHaveBeenCalledWith(
      "the-default",
      expect.any(Object),
    );

    mockModelFromPretrained.mockClear();
    setupModel([[0.5]]);
    await new CrossEncoderReranker(
      { model: "custom/model" },
      "the-default",
    ).rerank("q", ["a"]);
    expect(mockModelFromPretrained).toHaveBeenCalledWith(
      "custom/model",
      expect.any(Object),
    );
  });
});
