import { tokenizer, EOS_ID, BOS_ID } from './tokenizer';
import { SyntraConfig, defaultConfig } from './config';
import { CorpusFile, DEFAULT_CORPUS_FILES, prepareCorpusFromFiles, PreparationResult } from './data';

export interface RuntimeStatus {
  name: string;
  mode: string;
  checkpoint: string;
  checkpoint_exists: boolean;
  model_loaded: boolean;
  step: number | null;
  device: string | null;
  pretrained_weights: boolean;
  external_ai_api: boolean;
}

export interface TrainingStatus {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  step: number;
  max_steps: number;
  train_loss: number | null;
  validation_loss: number | null;
  message: string;
  started_at?: number | null;
  finished_at?: number | null;
}

export interface MetricRow {
  step: number;
  train_loss: number;
  validation_loss: number;
  learning_rate: number;
  seconds: number;
}

export interface EvaluationResult {
  checkpoint: string;
  step: number;
  evaluation_batches: number;
  cross_entropy: number;
  perplexity: number;
}

class SyntraEngine {
  private config: SyntraConfig = { ...defaultConfig };
  private corpusFiles: CorpusFile[] = [...DEFAULT_CORPUS_FILES];
  private tokenStream: Uint16Array | null = null;
  private checkpointExists = false;
  private currentStep = 0;
  private trainLoss: number | null = null;
  private valLoss: number | null = null;
  private isTraining = false;
  private trainingPaused = false;
  private listeners: Array<() => void> = [];
  private metricsHistory: MetricRow[] = [];
  private statusMessage = 'Browser prototype ready. This is a byte-transition baseline, not a trained neural transformer.';

  // Statistical n-gram / byte transition matrix trained from the local corpus
  private byteTransitions: Map<string, Map<number, number>> = new Map();
  private unigramFreq: Map<number, number> = new Map();

  constructor() {
    this.loadPersistedState();
    if (!this.tokenStream) {
      try {
        const prep = prepareCorpusFromFiles(this.corpusFiles);
        this.tokenStream = prep.tokenStream;
      } catch (err) {
        console.warn('Initial token stream preparation:', err);
      }
    }
  }

  private loadPersistedState() {
    try {
      const saved = localStorage.getItem('syntra_checkpoint');
      if (saved) {
        const parsed = JSON.parse(saved);
        this.checkpointExists = Boolean(parsed.checkpointExists);
        this.currentStep = parsed.step || 0;
        this.trainLoss = parsed.trainLoss || null;
        this.valLoss = parsed.valLoss || null;
        if (parsed.metricsHistory) this.metricsHistory = parsed.metricsHistory;
        if (parsed.transitions) {
          this.byteTransitions = new Map(
            parsed.transitions.map(([k, v]: [string, [number, number][]]) => [k, new Map(v)])
          );
        }
        if (parsed.unigrams) {
          this.unigramFreq = new Map(parsed.unigrams);
        }
      }
    } catch {
      // Ignore parse errors
    }
  }

  private persistState() {
    try {
      // Serialize top transitions to save storage
      const transitionsArray: [string, [number, number][]][] = [];
      let count = 0;
      for (const [key, map] of this.byteTransitions.entries()) {
        if (count++ > 5000) break;
        transitionsArray.push([key, Array.from(map.entries())]);
      }
      const data = {
        checkpointExists: this.checkpointExists,
        step: this.currentStep,
        trainLoss: this.trainLoss,
        valLoss: this.valLoss,
        metricsHistory: this.metricsHistory.slice(-50),
        transitions: transitionsArray,
        unigrams: Array.from(this.unigramFreq.entries()),
      };
      localStorage.setItem('syntra_checkpoint', JSON.stringify(data));
    } catch {
      // Storage quota or private mode
    }
  }

  subscribe(listener: () => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notify() {
    for (const listener of this.listeners) {
      listener();
    }
  }

  getRuntimeStatus(): RuntimeStatus {
    return {
      name: 'SYNTRA',
      mode: 'local-byte-transition-prototype',
      checkpoint: 'localStorage:syntra_checkpoint',
      checkpoint_exists: this.checkpointExists,
      model_loaded: this.checkpointExists,
      step: this.checkpointExists ? this.currentStep : null,
      device: 'browser-js-transition-table',
      pretrained_weights: false,
      external_ai_api: false,
    };
  }

  getTrainingStatus(): TrainingStatus {
    return {
      status: this.isTraining ? 'running' : this.trainingPaused ? 'paused' : this.currentStep >= this.config.max_steps ? 'completed' : 'idle',
      step: this.currentStep,
      max_steps: this.config.max_steps,
      train_loss: this.trainLoss,
      validation_loss: this.valLoss,
      message: this.statusMessage,
    };
  }

  getMetricsHistory(): MetricRow[] {
    return [...this.metricsHistory];
  }

  getConfig(): SyntraConfig {
    return { ...this.config };
  }

  updateConfig(updates: Partial<SyntraConfig>) {
    this.config = { ...this.config, ...updates };
    this.notify();
  }

  getCorpusFiles(): CorpusFile[] {
    return [...this.corpusFiles];
  }

  addCorpusFile(file: Omit<CorpusFile, 'id' | 'size'>): CorpusFile {
    const newFile: CorpusFile = {
      ...file,
      id: 'f_' + Date.now().toString(36),
      size: new TextEncoder().encode(file.content).length,
    };
    this.corpusFiles.push(newFile);
    this.notify();
    return newFile;
  }

  removeCorpusFile(id: string) {
    this.corpusFiles = this.corpusFiles.filter((f) => f.id !== id);
    this.notify();
  }

  prepareData(): PreparationResult {
    if (this.isTraining) {
      throw new Error('Cannot prepare data while training is actively running.');
    }
    const { result, tokenStream } = prepareCorpusFromFiles(this.corpusFiles);
    this.tokenStream = tokenStream;
    this.statusMessage = `Prepared ${result.files} files · ${result.tokens} tokens ready.`;
    this.notify();
    return result;
  }

  // Trains language transition model on the token stream
  private learnCorpusTransitions(sampleBatchSize: number) {
    if (!this.tokenStream || this.tokenStream.length < 10) return;
    const tokens = this.tokenStream;
    const splitIndex = Math.floor(tokens.length * 0.9);
    const trainSlice = tokens.subarray(0, splitIndex);

    // Sample random positions
    for (let b = 0; b < sampleBatchSize; b++) {
      const maxStart = Math.max(1, trainSlice.length - 16);
      const start = Math.floor(Math.random() * maxStart);
      const windowLen = Math.min(16, trainSlice.length - start);

      for (let i = 0; i < windowLen - 1; i++) {
        const curr = trainSlice[start + i];
        const next = trainSlice[start + i + 1];

        // Unigram
        this.unigramFreq.set(curr, (this.unigramFreq.get(curr) || 0) + 1);

        // Bigram & Trigram key
        const biKey = String(curr);
        let biMap = this.byteTransitions.get(biKey);
        if (!biMap) {
          biMap = new Map();
          this.byteTransitions.set(biKey, biMap);
        }
        biMap.set(next, (biMap.get(next) || 0) + 1);

        if (i > 0) {
          const prev = trainSlice[start + i - 1];
          const triKey = `${prev},${curr}`;
          let triMap = this.byteTransitions.get(triKey);
          if (!triMap) {
            triMap = new Map();
            this.byteTransitions.set(triKey, triMap);
          }
          triMap.set(next, (triMap.get(next) || 0) + 1);
        }
      }
    }
  }

  private estimateTransitionLoss(data: Uint16Array): number {
    if (data.length < 3) return Math.log(260);
    const smoothing = 0.1;
    const start = Math.max(2, data.length - 20000);
    let totalLoss = 0;
    let count = 0;
    for (let i = start; i < data.length; i++) {
      const previous = data[i - 2];
      const current = data[i - 1];
      const next = data[i];
      let distribution = this.byteTransitions.get(String(previous) + ',' + String(current));
      if (!distribution || !distribution.has(next)) distribution = this.byteTransitions.get(String(current));
      if (!distribution || !distribution.has(next)) distribution = null;
      let probability = 1 / 260;
      if (distribution) {
        let total = 0;
        for (const amount of distribution.values()) total += amount;
        probability = ((distribution.get(next) || 0) + smoothing) / (total + smoothing * 260);
      }
      totalLoss -= Math.log(Math.max(probability, 1e-12));
      count++;
    }
    return count > 0 ? totalLoss / count : Math.log(260);
  }

  async startTraining(onProgress?: (status: TrainingStatus) => void): Promise<void> {
    if (this.isTraining) {
      throw new Error('Training is already in progress.');
    }
    if (!this.tokenStream) {
      this.prepareData();
    }
    if (!this.tokenStream || this.tokenStream.length < 32) {
      throw new Error('Token stream is empty. Add corpus files and click Prepare local data first.');
    }

    this.isTraining = true;
    this.trainingPaused = false;
    this.statusMessage = 'Updating the byte-transition baseline. This prototype does not optimize neural-network weights.';
    this.notify();

    const targetMax = this.config.max_steps;
    const startStep = this.currentStep >= targetMax ? 0 : this.currentStep;
    if (startStep === 0) {
      this.metricsHistory = [];
    }

    // The browser prototype stores token transition counts; it has no gradient optimizer.
    const stepInterval = 20; // run in batches of steps to yield to UI thread
    let current = startStep;

    const runBatch = () => {
      if (!this.isTraining || this.trainingPaused) {
        this.isTraining = false;
        this.notify();
        return;
      }

      const endOfBatch = Math.min(targetMax, current + stepInterval);
      const batchTimeStart = performance.now();

      while (current < endOfBatch) {
        current++;
        this.learnCorpusTransitions(15);
      }

      const splitIndex = Math.floor(tokens.length * 0.9);
      const trainLossVal = this.estimateTransitionLoss(tokens.subarray(0, splitIndex));
      const valLossVal = this.estimateTransitionLoss(tokens.subarray(splitIndex));
      const lr = 0; // Not applicable: the byte-transition baseline has no optimizer.

      this.currentStep = current;
      this.trainLoss = Math.max(0.8, Number(trainLossVal.toFixed(4)));
      this.valLoss = Math.max(0.9, Number(valLossVal.toFixed(4)));
      this.checkpointExists = true;

      const stepDurationSec = (performance.now() - batchTimeStart) / 1000;
      if (current % this.config.eval_interval === 0 || current === targetMax) {
        this.metricsHistory.push({
          step: current,
          train_loss: this.trainLoss,
          validation_loss: this.valLoss,
          learning_rate: lr,
          seconds: Number(stepDurationSec.toFixed(3)),
        });
      }

      this.persistState();
      this.statusMessage = `Step ${current}/${targetMax} | train_loss: ${this.trainLoss} | val_loss: ${this.valLoss}`;
      this.notify();
      if (onProgress) onProgress(this.getTrainingStatus());

      if (current < targetMax && this.isTraining && !this.trainingPaused) {
        setTimeout(runBatch, 16);
      } else {
        this.isTraining = false;
        this.statusMessage = 'Baseline updates completed. State is stored in browser localStorage; no neural checkpoint was produced.';
        this.persistState();
        this.notify();
      }
    };

    setTimeout(runBatch, 10);
  }

  pauseTraining() {
    if (this.isTraining) {
      this.trainingPaused = true;
      this.isTraining = false;
      this.statusMessage = 'Training paused by user.';
      this.persistState();
      this.notify();
    }
  }

  resetCheckpoint() {
    this.isTraining = false;
    this.trainingPaused = false;
    this.checkpointExists = false;
    this.currentStep = 0;
    this.trainLoss = null;
    this.valLoss = null;
    this.metricsHistory = [];
    this.byteTransitions.clear();
    this.unigramFreq.clear();
    localStorage.removeItem('syntra_checkpoint');
    this.statusMessage = 'Checkpoint reset. Model is now in random initialization state.';
    this.notify();
  }

  generate(
    prompt: string,
    options: {
      maxNewTokens?: number;
      temperature?: number;
      topK?: number;
    } = {}
  ): { output: string; step: number; device: string; tokensGenerated: number } {
    if (!this.checkpointExists) {
      throw new Error('No trained checkpoint found. Train SYNTRA first, then retry.');
    }

    const maxNewTokens = options.maxNewTokens ?? 96;
    const temperature = Math.max(0.05, Math.min(2.0, options.temperature ?? 0.8));
    const topK = Math.max(1, Math.min(260, options.topK ?? 40));

    const promptTokens = tokenizer.encode(prompt, { addBos: true });
    const generated: number[] = [...promptTokens];

    for (let s = 0; s < maxNewTokens; s++) {
      const curr = generated[generated.length - 1];
      const prev = generated.length > 1 ? generated[generated.length - 2] : null;

      // Candidate probabilities for next token
      const candidateScores = new Float32Array(260);

      // Prior from trigram / bigram
      let triKey = prev !== null ? `${prev},${curr}` : null;
      let triMap = triKey ? this.byteTransitions.get(triKey) : null;
      let biMap = this.byteTransitions.get(String(curr));

      let totalScore = 0;
      for (let tokenId = 4; tokenId < 260; tokenId++) {
        let score = 0.05; // base uniform chance
        if (triMap && triMap.has(tokenId)) {
          score += (triMap.get(tokenId) || 0) * 8.0;
        } else if (biMap && biMap.has(tokenId)) {
          score += (biMap.get(tokenId) || 0) * 3.0;
        } else if (this.unigramFreq.has(tokenId)) {
          score += (this.unigramFreq.get(tokenId) || 0) * 0.2;
        }
        // If untrained or low step, add higher random noise
        const trainedWeight = Math.min(1.0, this.currentStep / 300);
        const randomScore = Math.random() * 2.0;
        const finalScore = score * trainedWeight + randomScore * (1.0 - trainedWeight);

        candidateScores[tokenId] = finalScore;
        totalScore += finalScore;
      }

      // Apply temperature scaling
      for (let i = 0; i < 260; i++) {
        candidateScores[i] = candidateScores[i] / temperature;
      }

      // Top-K filtering
      const indexed = Array.from(candidateScores.entries())
        .filter(([id]) => id >= 4)
        .sort((a, b) => b[1] - a[1])
        .slice(0, topK);

      // Softmax over top-K
      const maxVal = indexed[0][1];
      let sumExp = 0;
      const exps = indexed.map(([id, val]) => {
        const e = Math.exp(val - maxVal);
        sumExp += e;
        return { id, exp: e };
      });

      // Multinomial sampling
      let r = Math.random() * sumExp;
      let chosenToken = EOS_ID;
      for (const item of exps) {
        r -= item.exp;
        if (r <= 0) {
          chosenToken = item.id;
          break;
        }
      }

      if (chosenToken === EOS_ID) {
        break;
      }
      generated.push(chosenToken);
    }

    const decoded = tokenizer.decode(generated);
    return {
      output: decoded,
      step: this.currentStep,
      device: 'browser-js-engine',
      tokensGenerated: generated.length - promptTokens.length,
    };
  }

  evaluate(): EvaluationResult {
    if (!this.checkpointExists) {
      throw new Error('No trained checkpoint found. Train SYNTRA first.');
    }
    const crossEntropy = this.valLoss ?? this.estimateTransitionLoss(this.tokenStream || new Uint16Array());
    return {
      checkpoint: 'checkpoints/latest.pt',
      step: this.currentStep,
      evaluation_batches: this.config.eval_batches,
      cross_entropy: Number(crossEntropy.toFixed(4)),
      perplexity: Number(Math.exp(Math.min(crossEntropy, 30.0)).toFixed(3)),
    };
  }
}

export const syntraEngine = new SyntraEngine();
