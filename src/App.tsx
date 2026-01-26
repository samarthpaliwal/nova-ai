import { useEffect, useRef, useState } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';
import { ConnectionStatus } from './types';

type ChatMessage = { role: 'user' | 'ai'; text: string };

type IntakeData = {
  name: string;
  age: string;
  primaryConcern: string;
  goals: string;
  previousTherapy: string;
  preferredApproach: string;
  completed: boolean;
};

// Gemini 2.0 Flash is the ONLY model supporting Live streaming audio with @google/genai
// Gemini 3.0 does NOT support Live streaming as of January 2025
const LIVE_MODEL = 'models/gemini-2.0-flash-exp';

// Audio config
const IN_SAMPLE_RATE = 16000;
const OUT_SAMPLE_RATE = 24000;
const BUFFER_SIZE = 4096; // Larger chunks = fewer API calls

// Silence detection
const SILENCE_THRESHOLD = 0.008;
const SILENCE_MS = 1200;

// Timeouts
const CONNECT_TIMEOUT_MS = 15000;
const MIC_TIMEOUT_MS = 10000;

function now() {
  return new Date().toLocaleTimeString();
}

function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = window.setTimeout(() => reject(new Error(`${label} timeout`)), ms);
    p.then((v) => {
      clearTimeout(t);
      resolve(v);
    }).catch((e) => {
      clearTimeout(t);
      reject(e);
    });
  });
}

function computeRms(x: Float32Array) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
}

function floatToPcm16LE(input: Float32Array): Uint8Array {
  const out = new Uint8Array(input.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < input.length; i++) {
    let v = Math.max(-1, Math.min(1, input[i]));
    const s = v < 0 ? v * 0x8000 : v * 0x7fff;
    view.setInt16(i * 2, s, true);
  }
  return out;
}

function bytesToBase64(bytes: Uint8Array): string {
  let bin = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, Math.min(i + chunk, bytes.length)));
  }
  return btoa(bin);
}

// Decode base64 audio from AI response
async function decodeAiAudio(base64: string, ctx: AudioContext): Promise<AudioBuffer> {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return await ctx.decodeAudioData(bytes.buffer);
}

export default function App() {
  const API_KEY = (import.meta.env.VITE_GEMINI_API_KEY ?? '').trim();

  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [step, setStep] = useState('');
  const [lastEvent, setLastEvent] = useState('');
  const [errorText, setErrorText] = useState('');
  const [listening, setListening] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [showIntake, setShowIntake] = useState(false);
  const [intakeData, setIntakeData] = useState<IntakeData>(() => {
    const saved = localStorage.getItem('therapy_intake');
    return saved ? JSON.parse(saved) : {
      name: '',
      age: '',
      primaryConcern: '',
      goals: '',
      previousTherapy: '',
      preferredApproach: '',
      completed: false,
    };
  });

  // Session state
  const sessionRef = useRef<any>(null);
  const sessionStateRef = useRef<'closed' | 'opening' | 'open'>('closed');

  // Audio pipeline
  const inputCtxRef = useRef<AudioContext | null>(null);
  const outputCtxRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // Turn management
  const silenceTimerRef = useRef<number | null>(null);
  const inTurnRef = useRef(false);
  const cleanupCalledRef = useRef(false);

  // Memory
  const memoryRef = useRef(localStorage.getItem('therapy_memory') || '');

  const setEvent = (msg: string) => setLastEvent(`${now()} • ${msg}`);
  const setErr = (where: string, e: unknown) => {
    const msg = e instanceof Error ? e.message : typeof e === 'string' ? e : JSON.stringify(e);
    setErrorText(`${where}: ${msg}`);
    setEvent(`ERROR @ ${where}`);
    console.error(`[${where}]`, e);
  };

  const clearSilenceTimer = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  };

  const endUserTurn = () => {
    if (sessionStateRef.current !== 'open') return;
    if (!inTurnRef.current) return;

    inTurnRef.current = false;
    clearSilenceTimer();

    try {
      // The Live API requires BOTH messages to signal turn completion
      // 1) audioStreamEnd tells the model no more audio is coming
      sessionRef.current?.sendRealtimeInput?.({ audioStreamEnd: {} });
      
      // 2) turnComplete triggers the model to generate a response
      sessionRef.current?.sendRealtimeInput?.({ turnComplete: {} });
      
      setEvent('Turn ended → model responding');
    } catch (e) {
      // Swallow errors in turn completion - don't break audio pipeline
      console.warn('endUserTurn error:', e);
    }
  };

  const cleanup = () => {
    if (cleanupCalledRef.current) return;
    cleanupCalledRef.current = true;

    sessionStateRef.current = 'closed';
    setListening(false);
    clearSilenceTimer();
    inTurnRef.current = false;

    // Stop audio processor
    try {
      processorRef.current?.disconnect();
    } catch {}
    processorRef.current = null;

    try {
      sourceRef.current?.disconnect();
    } catch {}
    sourceRef.current = null;

    // Stop mic tracks
    try {
      micStreamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {}
    micStreamRef.current = null;

    // Close audio contexts
    try {
      inputCtxRef.current?.close();
    } catch {}
    inputCtxRef.current = null;

    try {
      outputCtxRef.current?.close();
    } catch {}
    outputCtxRef.current = null;

    // Close session
    try {
      sessionRef.current?.close?.();
    } catch {}
    sessionRef.current = null;

    setStep('');
    cleanupCalledRef.current = false;
  };

  const sendAudioChunk = (frameFloat32: Float32Array) => {
    // Circuit breaker: only send if session is fully open
    if (sessionStateRef.current !== 'open') return;
    if (!sessionRef.current?.sendRealtimeInput) return;

    try {
      const pcm16 = floatToPcm16LE(frameFloat32);
      const b64 = bytesToBase64(pcm16);

      // SDK expects 'realtimeInput' with 'mediaChunks' array containing audio data
      sessionRef.current.sendRealtimeInput({
        mediaChunks: [
          {
            data: b64,
            mimeType: `audio/pcm;rate=${IN_SAMPLE_RATE}`,
          },
        ],
      });
    } catch (e) {
      // Never throw in audio callback - just log
      console.warn('sendAudioChunk error:', e);
    }
  };

  const startMicPipeline = async () => {
    setStep('Setting up mic…');

    // Create audio contexts
    const inputCtx = new AudioContext({ sampleRate: IN_SAMPLE_RATE });
    const outputCtx = new AudioContext({ sampleRate: OUT_SAMPLE_RATE });

    // Resume contexts (required by browser security policy after user gesture)
    if (inputCtx.state === 'suspended') {
      await withTimeout(inputCtx.resume(), 2000, 'input resume');
    }
    if (outputCtx.state === 'suspended') {
      await withTimeout(outputCtx.resume(), 2000, 'output resume');
    }

    if (inputCtx.state !== 'running') {
      throw new Error('Input AudioContext failed to resume - user gesture required');
    }

    inputCtxRef.current = inputCtx;
    outputCtxRef.current = outputCtx;

    // Get mic stream
    const stream = await withTimeout(
      navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: IN_SAMPLE_RATE,
        },
      }),
      MIC_TIMEOUT_MS,
      'getUserMedia'
    );
    micStreamRef.current = stream;

    const source = inputCtx.createMediaStreamSource(stream);
    sourceRef.current = source;

    const processor = inputCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      // Circuit breaker: stop immediately if not connected
      if (sessionStateRef.current !== 'open') return;
      if (!listening) return;

      const ch0 = e.inputBuffer.getChannelData(0);
      const frame = new Float32Array(ch0);

      // Start turn on first audio
      if (!inTurnRef.current) {
        inTurnRef.current = true;
        setEvent('🎤 Streaming audio');
      }

      sendAudioChunk(frame);

      // Silence detection
      const rms = computeRms(frame);
      if (rms < SILENCE_THRESHOLD) {
        if (!silenceTimerRef.current) {
          silenceTimerRef.current = window.setTimeout(() => {
            setListening(false);
            endUserTurn();
          }, SILENCE_MS);
        }
      } else {
        clearSilenceTimer();
      }
    };

    source.connect(processor);
    processor.connect(inputCtx.destination);

    setStep('🎙️ Ready');
    setEvent('Mic pipeline ready');
  };

  const startSession = async () => {
    setErrorText('');

    if (!API_KEY) {
      setStatus(ConnectionStatus.ERROR);
      setErr('config', 'Missing VITE_GEMINI_API_KEY in .env');
      return;
    }

    // Check if intake questionnaire is completed
    if (!intakeData.completed) {
      setShowIntake(true);
      return;
    }

    cleanup();

    setStatus(ConnectionStatus.CONNECTING);
    sessionStateRef.current = 'opening';
    setStep('Connecting to Gemini…');
    setEvent('Initiating Live session');

    try {
      const ai = new GoogleGenAI({ apiKey: API_KEY });

      // Wait for session to fully open before starting mic
      await withTimeout(
        new Promise<void>((resolve, reject) => {
          ai.live
            .connect({
              model: LIVE_MODEL,
              config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                  voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
                },
                systemInstruction: {
                  parts: [
                    {
                      text: `You are a warm, empathetic therapy companion named Luna. Use simple, conversational language. Keep responses brief and supportive (2-3 sentences). Ask one gentle follow-up question.

CLIENT PROFILE:
- Name: ${intakeData.name}
- Age: ${intakeData.age}
- Primary Concern: ${intakeData.primaryConcern}
- Goals: ${intakeData.goals}
- Previous Therapy: ${intakeData.previousTherapy}
- Preferred Approach: ${intakeData.preferredApproach}

Use this information to personalize your responses. Reference their goals and concerns naturally. Adapt your therapeutic approach to their stated preferences.

CRITICAL SAFETY: If the user expresses feeling unsafe, mentions self-harm, or is in crisis:
- Respond with compassion: "I hear that you're struggling. You don't have to face this alone."
- Encourage: "Please reach out to someone you trust, or contact a crisis line. In the US: 988 Suicide & Crisis Lifeline."
- DO NOT provide graphic details or instructions about self-harm methods.

Session context: ${memoryRef.current || '(none)'}`,
                    },
                  ],
                },
              },
              callbacks: {
                onopen: () => {
                  sessionStateRef.current = 'open';
                  setStatus(ConnectionStatus.CONNECTED);
                  setStep('✓ Connected');
                  setEvent('WebSocket OPEN');
                  resolve();
                },
                onmessage: async (msg: LiveServerMessage) => {
                  try {
                    // Extract text transcription from serverContent.turnComplete
                    const turnComplete = (msg as any)?.serverContent?.turnComplete;
                    if (turnComplete) {
                      // SDK returns text in outputTranscription field
                      const transcript = (msg as any)?.serverContent?.modelTurn?.parts?.[0]?.text;
                      if (transcript?.trim()) {
                        setMessages((prev) => [...prev, { role: 'ai', text: transcript.trim() }]);
                        
                        // Update memory
                        const updated = (memoryRef.current ? `${memoryRef.current}\n${transcript}` : transcript).slice(-4000);
                        memoryRef.current = updated;
                        localStorage.setItem('therapy_memory', updated);
                        
                        setEvent('AI responded');
                      }
                    }

                    // Play audio response
                    const parts = (msg as any)?.serverContent?.modelTurn?.parts ?? [];
                    for (const part of parts) {
                      if (part.inlineData?.mimeType?.startsWith('audio/') && part.inlineData?.data) {
                        if (outputCtxRef.current && outputCtxRef.current.state === 'running') {
                          try {
                            const buffer = await decodeAiAudio(part.inlineData.data, outputCtxRef.current);
                            const src = outputCtxRef.current.createBufferSource();
                            src.buffer = buffer;
                            src.connect(outputCtxRef.current.destination);
                            src.start();
                          } catch (e) {
                            console.warn('Audio playback error:', e);
                          }
                        }
                      }
                    }
                  } catch (e) {
                    console.warn('onmessage error:', e);
                  }
                },
                onclose: (e: any) => {
                  sessionStateRef.current = 'closed';
                  const code = e?.code ?? 'unknown';
                  const reason = e?.reason ?? '';
                  setStatus(ConnectionStatus.ERROR);
                  setErr('onclose', `WebSocket closed (${code}) ${reason}`);
                  cleanup();
                },
                onerror: (e: any) => {
                  sessionStateRef.current = 'closed';
                  setStatus(ConnectionStatus.ERROR);
                  setErr('onerror', e?.message || JSON.stringify(e));
                  cleanup();
                },
              },
            })
            .then((session) => {
              sessionRef.current = session;
            })
            .catch(reject);
        }),
        CONNECT_TIMEOUT_MS,
        'Live session connect'
      );

      // Only start mic AFTER session is fully open
      await startMicPipeline();
    } catch (e) {
      sessionStateRef.current = 'closed';
      setStatus(ConnectionStatus.ERROR);
      setErr('startSession', e);
      cleanup();
    }
  };

  const endSession = () => {
    setEvent('Session ended');
    cleanup();
    setStatus(ConnectionStatus.DISCONNECTED);
  };

  const importTextFile = async (file: File) => {
    try {
      const text = await file.text();
      const clipped = text.slice(-12000);
      memoryRef.current = clipped;
      localStorage.setItem('therapy_memory', clipped);
      setEvent('Context imported');
    } catch (e) {
      setErr('import', e);
    }
  };

  const resetMemory = () => {
    memoryRef.current = '';
    localStorage.removeItem('therapy_memory');
    setMessages([]);
    setEvent('Memory cleared');
  };

  const saveIntakeData = () => {
    const completed = { ...intakeData, completed: true };
    setIntakeData(completed);
    localStorage.setItem('therapy_intake', JSON.stringify(completed));
    setShowIntake(false);
    setEvent('Intake completed');
  };

  const updateIntakeField = (field: keyof IntakeData, value: string) => {
    setIntakeData(prev => ({ ...prev, [field]: value }));
  };

  const resetIntake = () => {
    const empty: IntakeData = {
      name: '',
      age: '',
      primaryConcern: '',
      goals: '',
      previousTherapy: '',
      preferredApproach: '',
      completed: false,
    };
    setIntakeData(empty);
    localStorage.removeItem('therapy_intake');
    setShowIntake(false);
    setEvent('Intake reset');
  };

  useEffect(() => {
    return () => cleanup();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const statusDot =
    status === ConnectionStatus.CONNECTED
      ? 'bg-emerald-500 shadow-[0_0_12px_#10b981]'
      : status === ConnectionStatus.ERROR
      ? 'bg-red-500 shadow-[0_0_12px_#ef4444]'
      : status === ConnectionStatus.CONNECTING
      ? 'bg-yellow-500 shadow-[0_0_12px_#eab308]'
      : 'bg-white/20';

  // Intake form validation
  const intakeValid = intakeData.name.trim() && 
                       intakeData.age.trim() && 
                       intakeData.primaryConcern.trim() && 
                       intakeData.goals.trim();

  return (
    <div className="w-full h-screen bg-[#0a0a0a] text-white flex flex-col items-center overflow-hidden font-sans">
      {/* Intake Modal */}
      {showIntake && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="bg-[#1a1a1a] border border-white/10 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-8">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-light mb-1">Welcome to Therapy</h2>
                <p className="text-white/50 text-sm">Help us personalize your experience</p>
              </div>
              <button
                onClick={() => setShowIntake(false)}
                className="text-white/40 hover:text-white/80 transition text-2xl leading-none"
              >
                ×
              </button>
            </div>

            <div className="space-y-6">
              {/* Name */}
              <div>
                <label className="block text-sm text-white/60 mb-2">What should we call you?</label>
                <input
                  type="text"
                  value={intakeData.name}
                  onChange={(e) => updateIntakeField('name', e.target.value)}
                  placeholder="Your preferred name"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-emerald-500/50 transition"
                />
              </div>

              {/* Age */}
              <div>
                <label className="block text-sm text-white/60 mb-2">Age range</label>
                <select
                  value={intakeData.age}
                  onChange={(e) => updateIntakeField('age', e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-emerald-500/50 transition"
                >
                  <option value="">Select age range</option>
                  <option value="18-24">18-24</option>
                  <option value="25-34">25-34</option>
                  <option value="35-44">35-44</option>
                  <option value="45-54">45-54</option>
                  <option value="55-64">55-64</option>
                  <option value="65+">65+</option>
                </select>
              </div>

              {/* Primary Concern */}
              <div>
                <label className="block text-sm text-white/60 mb-2">What brings you here today?</label>
                <textarea
                  value={intakeData.primaryConcern}
                  onChange={(e) => updateIntakeField('primaryConcern', e.target.value)}
                  placeholder="e.g., anxiety, stress, relationship issues, life transitions..."
                  rows={3}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-emerald-500/50 transition resize-none"
                />
              </div>

              {/* Goals */}
              <div>
                <label className="block text-sm text-white/60 mb-2">What would you like to achieve?</label>
                <textarea
                  value={intakeData.goals}
                  onChange={(e) => updateIntakeField('goals', e.target.value)}
                  placeholder="e.g., better sleep, manage stress, improve relationships, find clarity..."
                  rows={3}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-emerald-500/50 transition resize-none"
                />
              </div>

              {/* Previous Therapy */}
              <div>
                <label className="block text-sm text-white/60 mb-2">Have you tried therapy before?</label>
                <select
                  value={intakeData.previousTherapy}
                  onChange={(e) => updateIntakeField('previousTherapy', e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-emerald-500/50 transition"
                >
                  <option value="">Select option</option>
                  <option value="no">No, this is my first time</option>
                  <option value="yes-helpful">Yes, and it was helpful</option>
                  <option value="yes-not-helpful">Yes, but it wasn't for me</option>
                  <option value="prefer-not-say">Prefer not to say</option>
                </select>
              </div>

              {/* Preferred Approach */}
              <div>
                <label className="block text-sm text-white/60 mb-2">Preferred conversation style</label>
                <select
                  value={intakeData.preferredApproach}
                  onChange={(e) => updateIntakeField('preferredApproach', e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-emerald-500/50 transition"
                >
                  <option value="">Select approach</option>
                  <option value="gentle-exploratory">Gentle and exploratory</option>
                  <option value="solution-focused">Solution-focused and practical</option>
                  <option value="reflective">Reflective and thought-provoking</option>
                  <option value="supportive">Supportive and validating</option>
                </select>
              </div>
            </div>

            {/* Submit */}
            <div className="mt-8 flex gap-3">
              <button
                onClick={saveIntakeData}
                disabled={!intakeValid}
                className={`flex-1 py-3 rounded-lg font-medium transition ${
                  intakeValid
                    ? 'bg-emerald-500 hover:bg-emerald-600 text-white'
                    : 'bg-white/5 text-white/30 cursor-not-allowed'
                }`}
              >
                Begin Session
              </button>
              {intakeData.completed && (
                <button
                  onClick={() => setShowIntake(false)}
                  className="px-6 py-3 rounded-lg border border-white/10 hover:bg-white/5 transition"
                >
                  Cancel
                </button>
              )}
            </div>

            {intakeData.completed && (
              <button
                onClick={resetIntake}
                className="w-full mt-3 text-xs text-white/40 hover:text-white/60 transition"
              >
                Reset questionnaire
              </button>
            )}
          </div>
        </div>
      )}

      {/* Header */}
      <div className="w-full max-w-2xl px-6 pt-6 pb-4 flex items-center justify-between border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${statusDot} transition-all`} />
          <span className="text-sm text-white/60">{status}</span>
          {step && <span className="text-xs text-white/40">• {step}</span>}
        </div>
        <div className="flex items-center gap-2">
          {intakeData.completed && (
            <button
              onClick={() => setShowIntake(true)}
              className="text-[11px] px-3 py-1.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 transition"
              title="View/edit client profile"
            >
              👤 Profile
            </button>
          )}
          <label className="text-[11px] px-3 py-1.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 transition cursor-pointer">
            Import context
            <input
              type="file"
              accept=".txt"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void importTextFile(f);
                e.currentTarget.value = '';
              }}
            />
          </label>
          <button
            onClick={resetMemory}
            className="text-[11px] px-3 py-1.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 transition"
          >
            Clear memory
          </button>
          <button
            onClick={status === ConnectionStatus.CONNECTED ? endSession : startSession}
            className="text-[11px] px-3 py-1.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 transition"
          >
            {status === ConnectionStatus.CONNECTED ? 'End' : 'Start'}
          </button>
        </div>
      </div>

      {/* Event log */}
      {lastEvent && (
        <div className="w-full max-w-2xl px-6 py-2 text-[10px] text-white/30 font-mono">
          {lastEvent}
        </div>
      )}

      {/* Error banner */}
      {errorText && (
        <div className="w-full max-w-2xl px-6 py-3">
          <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
            ⚠️ {errorText}
          </div>
        </div>
      )}

      {/* Chat feed */}
      <div className="flex-1 w-full max-w-2xl px-6 py-6 overflow-y-auto flex flex-col gap-4">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center">
            <div className="text-4xl mb-4">🌙</div>
            <h1 className="text-2xl mb-2 text-white/90">
              {intakeData.completed ? `Welcome back, ${intakeData.name}` : 'Welcome'}
            </h1>
            <p className="text-white/40 text-sm mb-4">
              {intakeData.completed
                ? "Hold the mic when you're ready to talk."
                : 'Complete the questionnaire to begin.'}
            </p>
            {!intakeData.completed && (
              <button
                onClick={() => setShowIntake(true)}
                className="px-6 py-3 rounded-full bg-emerald-500 hover:bg-emerald-600 transition font-medium"
              >
                Start Questionnaire
              </button>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {messages.slice(-18).map((m, i) => (
              <div
                key={i}
                className={`px-4 py-3 rounded-2xl max-w-[85%] ${
                  m.role === 'ai'
                    ? 'bg-white/5 self-start'
                    : 'bg-emerald-500/10 self-end text-right'
                }`}
              >
                <div className="text-[10px] text-white/40 mb-1">
                  {m.role === 'ai' ? '🌙 Luna' : '💭 You'}
                </div>
                <div className="text-sm leading-relaxed text-white/90">{m.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Bottom instruction */}
      <div className="w-full max-w-2xl px-6 py-2 text-center text-[10px] text-white/30">
        Hold to talk • Release to send • Silence also sends
      </div>

      {/* Mic button */}
      <div className="w-full max-w-2xl px-6 pb-8 flex justify-center">
        <button
          disabled={status !== ConnectionStatus.CONNECTED}
          onMouseDown={() => {
            setListening(true);
            clearSilenceTimer();
            setEvent('🎤 Listening');
          }}
          onMouseUp={() => {
            setListening(false);
            endUserTurn();
          }}
          onTouchStart={() => {
            setListening(true);
            clearSilenceTimer();
            setEvent('🎤 Listening');
          }}
          onTouchEnd={() => {
            setListening(false);
            endUserTurn();
          }}
          className={`w-20 h-20 rounded-full flex items-center justify-center text-3xl transition-all ${
            status !== ConnectionStatus.CONNECTED
              ? 'bg-[#1e1e1e] opacity-40 cursor-not-allowed'
              : listening
              ? 'bg-emerald-500 shadow-[0_0_40px_#10b981] scale-110'
              : 'bg-[#2a2a2a] hover:bg-[#303030] hover:scale-105'
          }`}
          aria-label="Hold to speak"
        >
          🎙️
        </button>
      </div>
    </div>
  );
}