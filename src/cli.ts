#!/usr/bin/env tsx

import arg from 'arg';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { AudioContext, OfflineAudioContext } from 'node-web-audio-api';
import { pipeline } from '@huggingface/transformers';
import { levenshteinDistance } from './stringMatching';
import { mergeOverlappingBleeps } from './bleepMerger';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '..');
const TMP_DIR = path.join(PROJECT_ROOT, 'tmp');
const execFileAsync = promisify(execFile);

const DEFAULT_MODEL = 'Xenova/whisper-small.en';
const DEFAULT_REVISION = 'main';
const DEFAULT_WORDS = 'fuck,shit,damn,bitch,dick,cunt,bastard,asshole';

function audioBufferToWav(buffer: AudioBuffer): Buffer {
  const numberOfChannels = buffer.numberOfChannels;
  const length = buffer.length * numberOfChannels * 2 + 44;
  const outputBuffer = new ArrayBuffer(length);
  const view = new DataView(outputBuffer);
  const channels: Float32Array[] = [];
  let offset = 0;
  let pos = 0;

  const setUint16 = (data: number) => { view.setUint16(pos, data, true); pos += 2; };
  const setUint32 = (data: number) => { view.setUint32(pos, data, true); pos += 4; };

  setUint32(0x46464952);
  setUint32(length - 8);
  setUint32(0x45564157);
  setUint32(0x20746d66);
  setUint32(16);
  setUint16(1);
  setUint16(numberOfChannels);
  setUint32(buffer.sampleRate);
  setUint32(buffer.sampleRate * 2 * numberOfChannels);
  setUint16(numberOfChannels * 2);
  setUint16(16);
  setUint32(0x61746164);
  setUint32(length - pos - 4);

  for (let i = 0; i < numberOfChannels; i++) channels.push(buffer.getChannelData(i));

  while (pos < length) {
    for (let i = 0; i < numberOfChannels; i++) {
      const sample = Math.max(-1, Math.min(1, channels[i][offset]));
      view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      pos += 2;
    }
    offset++;
  }

  return Buffer.from(outputBuffer);
}

function help() {
  console.log(`
Usage: npx tsx src/cli.ts --input <file> [options]

Options:
  -i, --input <path>           Input audio/video file (required)
  -o, --output <path>          Output file path (default: input-censored.ext)
  --words <list>               Comma-separated words to censor
                                (default: ${DEFAULT_WORDS})
  --bleep-sound <name>         Bleep sound: bleep, brown, dolphin, trex, silence
                               (default: bleep)
  --bleep-volume <0-100>       Bleep volume percentage (default: 80)
  --buffer <seconds>           Extra time buffer around each bleep (default: 0)
  --original-volume <0.0-1.0>  Original audio volume during bleeps (default: 0.0)
  --model <id>                 Whisper model (default: ${DEFAULT_MODEL})
  --language <code>            Language code (default: en)
  --match-mode <mode>          Match mode: exact, partial, fuzzy (default: exact)
  --fuzzy-distance <n>        Max Levenshtein distance for fuzzy (default: 1)
  --transcribe-only            Only transcribe, print result, don't bleep
  -h, --help                   Show this help
`);
}

async function ffmpeg(args: string[]): Promise<void> {
  const { stderr } = await execFileAsync('ffmpeg', ['-y', ...args]);
  if (stderr && stderr.includes('Error')) {
    throw new Error(`ffmpeg error: ${stderr}`);
  }
}

async function hasSubtitleStream(filePath: string): Promise<boolean> {
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v', 'error',
      '-select_streams', 's',
      '-show_entries', 'stream=index',
      '-of', 'csv=p=0',
      filePath,
    ]);
    return stdout.trim().length > 0;
  } catch {
    return false;
  }
}

async function main() {
  const parsed = arg({
    '--input': String,
    '--output': String,
    '--words': String,
    '--bleep-sound': String,
    '--bleep-volume': Number,
    '--buffer': Number,
    '--original-volume': Number,
    '--model': String,
    '--language': String,
    '--match-mode': String,
    '--fuzzy-distance': Number,
    '--transcribe-only': Boolean,
    '--help': Boolean,
    '-i': '--input',
    '-o': '--output',
    '-h': '--help',
  });

  if (parsed['--help'] || !parsed['--input']) {
    help();
    process.exit(parsed['--help'] ? 0 : 1);
  }

  const inputPath = parsed['--input'];
  const outputPath = parsed['--output'];
  const wordsStr = parsed['--words'] || DEFAULT_WORDS;
  const bleepSound = parsed['--bleep-sound'] || 'bleep';
  const bleepVolume = (parsed['--bleep-volume'] ?? 80) / 100;
  const buffer = parsed['--buffer'] ?? 0;
  const originalVolume = parsed['--original-volume'] ?? 0.0;
  const model = parsed['--model'] || DEFAULT_MODEL;
  const language = parsed['--language'] || 'en';
  const matchMode = parsed['--match-mode'] || 'exact';
  const fuzzyDistance = parsed['--fuzzy-distance'] ?? 1;
  const transcribeOnly = parsed['--transcribe-only'] ?? false;

  const words = wordsStr.split(',').map(w => w.trim().toLowerCase()).filter(Boolean);
  const ext = path.extname(inputPath).toLowerCase();
  const isVideo = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'].includes(ext);
  const isAudio = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'].includes(ext);

  if (!isVideo && !isAudio) {
    console.error(`Unsupported file type: ${ext}`);
    process.exit(1);
  }

  const defaultOutput = isVideo
    ? inputPath.replace(ext, `-censored${ext}`)
    : inputPath.replace(ext, '-censored.wav');
  const finalOutput = outputPath || defaultOutput;

  console.error(`Input: ${inputPath}`);
  console.error(`Output: ${finalOutput}`);
  console.error(`Words: ${words.join(', ')}`);
  console.error(`Bleep sound: ${bleepSound}`);
  console.error(`Bleep volume: ${Math.round(bleepVolume * 100)}%`);
  console.error(`Buffer: ${buffer}s`);
  console.error(`Model: ${model}`);
  console.error(`Language: ${language}`);
  console.error(`Match mode: ${matchMode}`);
  console.error('');

  await fs.mkdir(TMP_DIR, { recursive: true });

  const inputExt = ext;
  const tmpInput = path.join(TMP_DIR, `input${inputExt}`);
  const tmpAudio16k = path.join(TMP_DIR, 'audio_16k.wav');
  const tmpAudioOrig = path.join(TMP_DIR, 'audio_orig.wav');
  const tmpCensored = path.join(TMP_DIR, 'censored.wav');
  const tmpOutput = path.join(TMP_DIR, `output${inputExt}`);

  await fs.copyFile(inputPath, tmpInput);

  let audioArrayBuffer: ArrayBuffer;
  let isVideoInput = isVideo;

  if (isVideo) {
    console.error('[1/5] Extracting audio from video...');
    await ffmpeg(['-i', tmpInput, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', tmpAudio16k]);
    await ffmpeg(['-i', tmpInput, '-vn', '-acodec', 'pcm_s16le', '-ac', '2', tmpAudioOrig]);
    const data = await fs.readFile(tmpAudio16k);
    audioArrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  } else {
    const data = await fs.readFile(tmpInput);
    audioArrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }

  console.error('[2/5] Transcribing audio...');
  const audioCtx = new AudioContext();
  let audioBuffer: AudioBuffer;
  try {
    audioBuffer = await audioCtx.decodeAudioData(audioArrayBuffer);
  } finally {
    await audioCtx.close();
  }

  const mono = audioBuffer.numberOfChannels === 1
    ? audioBuffer.getChannelData(0)
    : (() => {
        const ch0 = audioBuffer.getChannelData(0);
        const ch1 = audioBuffer.getChannelData(1);
        const m = new Float32Array(audioBuffer.length);
        for (let i = 0; i < audioBuffer.length; i++) m[i] = (ch0[i] + ch1[i]) / 2;
        return m;
      })();

  let pcm16k: Float32Array;
  if (audioBuffer.sampleRate === 16000) {
    pcm16k = mono;
  } else {
    const offlineCtx = new OfflineAudioContext(1, Math.ceil((mono.length * 16000) / audioBuffer.sampleRate), 16000);
    const buf = offlineCtx.createBuffer(1, mono.length, audioBuffer.sampleRate);
    buf.copyToChannel(new Float32Array(mono), 0);
    const src = offlineCtx.createBufferSource();
    src.buffer = buf;
    src.connect(offlineCtx.destination);
    src.start();
    const rendered = await offlineCtx.startRendering();
    pcm16k = rendered.getChannelData(0);
  }

  const transcriber = await pipeline('automatic-speech-recognition', model, {
    revision: DEFAULT_REVISION,
    progress_callback: (p: any) => {
      if (p?.progress !== undefined) {
        const v = p.progress > 1 ? p.progress / 100 : p.progress;
        console.error(`  Model loading: ${Math.round(v * 100)}%`);
      }
    },
  });

  const opts: any = { chunk_length_s: 20, stride_length_s: 3, return_timestamps: 'word' };
  if (!model.includes('.en')) {
    opts.language = language;
    opts.task = 'transcribe';
  }

  const result = await transcriber(pcm16k, opts);
  const finalResult = Array.isArray(result)
    ? { text: result.map(r => r.text || '').join(' '), chunks: result.flatMap(r => r.chunks || []) }
    : result;

  const validChunks = (finalResult.chunks || []).filter((c: any) => c.timestamp && c.timestamp[0] !== null && c.timestamp[1] !== null);
  const nullCount = (finalResult.chunks || []).length - validChunks.length;
  if (nullCount > 0) console.error(`  Warning: ${nullCount} chunks had null timestamps (filtered out)`);

  console.error(`  Transcribed ${validChunks.length} words`);

  if (transcribeOnly) {
    console.log(JSON.stringify({ text: finalResult.text, chunks: validChunks }, null, 2));
    return;
  }

  console.error('[3/5] Matching words...');
  const matchedIndices = new Set<number>();
  validChunks.forEach((chunk: any, idx: number) => {
    const text = (chunk.text as string).toLowerCase().trim().replace(/[.,!?;:'"]/g, '');
    for (const word of words) {
      if (matchMode === 'exact' && text === word) { matchedIndices.add(idx); break; }
      if (matchMode === 'partial' && text.includes(word)) { matchedIndices.add(idx); break; }
      if (matchMode === 'fuzzy' && levenshteinDistance(text, word) <= fuzzyDistance) { matchedIndices.add(idx); break; }
    }
  });

  if (matchedIndices.size === 0) {
    console.error('No matches found. Try a different match mode or word list.');
    process.exit(1);
  }

  console.error(`  Matched ${matchedIndices.size} words`);

  const rawSegments = Array.from(matchedIndices).map(idx => {
    const c = validChunks[idx];
    return {
      word: c.text,
      start: Math.max(0, c.timestamp[0] - buffer),
      end: c.timestamp[1] + buffer,
    };
  });
  const finalSegments = mergeOverlappingBleeps(rawSegments);

  console.error(`[4/5] Applying ${finalSegments.length} bleeps...`);

  let bleepBuffer: AudioBuffer | null = null;
  if (bleepSound !== 'silence') {
    const bleepPath = path.join(PROJECT_ROOT, 'bleeps', `${bleepSound}.mp3`);
    const bleepData = await fs.readFile(bleepPath);
    const bleepCtx = new AudioContext();
    try {
      bleepBuffer = await bleepCtx.decodeAudioData(bleepData.buffer.slice(bleepData.byteOffset, bleepData.byteOffset + bleepData.byteLength));
    } finally {
      await bleepCtx.close();
    }
  }

  let originalAudioBuffer: AudioBuffer;
  if (isVideoInput) {
    const origData = await fs.readFile(tmpAudioOrig);
    const origAB = origData.buffer.slice(origData.byteOffset, origData.byteOffset + origData.byteLength);
    const origCtx = new AudioContext();
    try {
      originalAudioBuffer = await origCtx.decodeAudioData(origAB);
    } finally {
      await origCtx.close();
    }
  } else {
    const origCtx = new AudioContext();
    try {
      originalAudioBuffer = await origCtx.decodeAudioData(audioArrayBuffer);
    } finally {
      await origCtx.close();
    }
  }

  const offlineCtx = new OfflineAudioContext(
    originalAudioBuffer.numberOfChannels,
    originalAudioBuffer.length,
    originalAudioBuffer.sampleRate
  );

  const source = offlineCtx.createBufferSource();
  source.buffer = originalAudioBuffer;

  const gainNode = offlineCtx.createGain();
  source.connect(gainNode);
  gainNode.connect(offlineCtx.destination);

  gainNode.gain.setValueAtTime(1, offlineCtx.currentTime);

  for (const seg of finalSegments) {
    const dur = seg.end - seg.start;
    gainNode.gain.setValueAtTime(1, Math.max(0, seg.start - 0.01));
    gainNode.gain.linearRampToValueAtTime(originalVolume, seg.start);
    gainNode.gain.setValueAtTime(originalVolume, seg.end);
    gainNode.gain.linearRampToValueAtTime(1, seg.end + 0.01);

    if (bleepBuffer) {
      const bs = offlineCtx.createBufferSource();
      bs.buffer = bleepBuffer;
      const bg = offlineCtx.createGain();
      bg.gain.value = bleepVolume;
      bs.connect(bg);
      bg.connect(offlineCtx.destination);
      if (dur > bleepBuffer.duration) { bs.loop = true; bs.loopEnd = bleepBuffer.duration; }
      bs.start(seg.start, 0, dur);
    }
  }

  source.start(0);
  const rendered = await offlineCtx.startRendering();
  const wavBuffer = audioBufferToWav(rendered);

if (isVideoInput) {
    console.error('[5/5] Remuxing video with censored audio...');
    await fs.writeFile(tmpCensored, wavBuffer);

    const hasSubtitles = await hasSubtitleStream(tmpInput);
    const remuxArgs = ['-i', tmpInput, '-i', tmpCensored, '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-map', '0:v:0', '-map', '1:a:0'];
    if (hasSubtitles) {
      console.error('  Preserving subtitles...');
      remuxArgs.push('-c:s', 'copy', '-map', '0:s');
    }
    remuxArgs.push('-async', '1', tmpOutput);
    await ffmpeg(remuxArgs);
    await fs.copyFile(tmpOutput, finalOutput);
  } else {
    await fs.writeFile(finalOutput, wavBuffer);
  }

  for (const f of [tmpInput, tmpAudio16k, tmpAudioOrig, tmpCensored, tmpOutput]) {
    try { await fs.unlink(f); } catch {}
  }

  console.error(`Done! Output: ${finalOutput}`);
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});