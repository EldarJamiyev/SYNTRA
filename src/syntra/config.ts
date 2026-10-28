export interface SyntraConfig {
  seed: number;
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
  dropout: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  max_steps: number;
  eval_interval: number;
  eval_batches: number;
  grad_clip: number;
  num_threads: number;
  device: string;
  data_file: string;
  checkpoint_dir: string;
}

export const defaultConfig: SyntraConfig = {
  seed: 42,
  vocab_size: 260,
  block_size: 128,
  n_layer: 4,
  n_head: 6,
  n_embd: 384,
  dropout: 0.1,
  batch_size: 4,
  learning_rate: 0.0003,
  weight_decay: 0.1,
  max_steps: 1000,
  eval_interval: 100,
  eval_batches: 10,
  grad_clip: 1.0,
  num_threads: 4,
  device: "local-client",
  data_file: "data/processed/tokens.bin",
  checkpoint_dir: "checkpoints"
};

export function calculateParameterCount(config: SyntraConfig): {
  total: number;
  embeddings: number;
  layers: number;
  lmHead: number;
} {
  const tokenEmbed = config.vocab_size * config.n_embd;
  const posEmbed = config.block_size * config.n_embd;
  const embeddings = tokenEmbed + posEmbed;

  // Each transformer block:
  // LayerNorm 1: 2 * n_embd (gamma, beta)
  // Attention:
  //   qkv: (n_embd * 3 * n_embd) + 3 * n_embd
  //   projection: (n_embd * n_embd) + n_embd
  // LayerNorm 2: 2 * n_embd
  // FeedForward:
  //   fc1: (n_embd * 4 * n_embd) + 4 * n_embd
  //   fc2: (4 * n_embd * n_embd) + n_embd
  const qkvWeights = config.n_embd * 3 * config.n_embd + 3 * config.n_embd;
  const projWeights = config.n_embd * config.n_embd + config.n_embd;
  const ffwWeights = (config.n_embd * 4 * config.n_embd + 4 * config.n_embd) + (4 * config.n_embd * config.n_embd + config.n_embd);
  const lnWeights = 4 * config.n_embd;
  const perBlock = qkvWeights + projWeights + ffwWeights + lnWeights;
  const allBlocks = perBlock * config.n_layer;

  const finalNorm = 2 * config.n_embd;
  // Weight tying: lm_head shares weights with token_embedding, so 0 additional weights or unshared
  const lmHead = 0; // weight tied to token_embedding as per SyntraModel

  const total = embeddings + allBlocks + finalNorm + lmHead;

  return {
    total,
    embeddings,
    layers: allBlocks,
    lmHead,
  };
}
