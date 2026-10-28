/**
 * Lossless UTF-8 byte tokenizer for SYNTRA.
 * Token IDs 0-3 are reserved special tokens.
 * UTF-8 bytes 0-255 map to token IDs 4-259 (BYTE_OFFSET = 4).
 * Vocabulary size is fixed to 260.
 */

export const PAD_ID = 0;
export const BOS_ID = 1;
export const EOS_ID = 2;
export const UNK_ID = 3;
export const BYTE_OFFSET = 4;
export const VOCAB_SIZE = 260;

export class ByteTokenizer {
  readonly vocabSize: number = VOCAB_SIZE;

  encode(text: string, options: { addBos?: boolean; addEos?: boolean } = {}): number[] {
    const { addBos = true, addEos = false } = options;
    const tokens: number[] = [];
    if (addBos) tokens.push(BOS_ID);

    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    for (let i = 0; i < bytes.length; i++) {
      tokens.push(bytes[i] + BYTE_OFFSET);
    }

    if (addEos) tokens.push(EOS_ID);
    return tokens;
  }

  decode(tokenIds: number[]): string {
    const bytes: number[] = [];
    for (const tokenId of tokenIds) {
      if (tokenId >= BYTE_OFFSET && tokenId < VOCAB_SIZE) {
        bytes.push(tokenId - BYTE_OFFSET);
      }
    }
    const decoder = new TextDecoder('utf-8', { fatal: false });
    return decoder.decode(new Uint8Array(bytes));
  }

  tokenToString(tokenId: number): string {
    if (tokenId === PAD_ID) return '<PAD>';
    if (tokenId === BOS_ID) return '<BOS>';
    if (tokenId === EOS_ID) return '<EOS>';
    if (tokenId === UNK_ID) return '<UNK>';
    if (tokenId >= BYTE_OFFSET && tokenId < VOCAB_SIZE) {
      const val = tokenId - BYTE_OFFSET;
      if (val >= 32 && val <= 126) {
        return String.fromCharCode(val);
      }
      return '0x' + val.toString(16).padStart(2, '0');
    }
    throw new Error(`Token ID ${tokenId} outside vocabulary`);
  }
}

export const tokenizer = new ByteTokenizer();
