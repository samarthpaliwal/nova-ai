# Nova AI

Nova AI is a browser-based voice companion built with React, TypeScript and Vite. It opens a Gemini Live session over WebSocket, streams microphone audio to the model, plays back the spoken reply, and shows a transcript of the model's turns. The assistant persona ("Luna") is configured as a supportive, conversational companion that is personalised from a short intake questionnaire. It is a personal project, not a medical or mental-health service.

## Requirements

- Node.js 18 or newer and npm
- A modern desktop browser with microphone access (the app uses the Web Audio API and `getUserMedia`)
- A Google Gemini API key with access to the Live API model `gemini-2.0-flash-exp`

## Setup

```bash
git clone https://github.com/samarthpaliwal/nova-ai.git
cd nova-ai
npm install
```

The app reads one environment variable at build time:

| Variable | Purpose |
|---|---|
| `VITE_GEMINI_API_KEY` | Your Gemini API key. Read in `src/App.tsx` via `import.meta.env`. |

Create a local `.env` file in the project root (copy `.env.example`) and put your own key in it:

```env
# .env.example
VITE_GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file must stay local and must not be committed. Add `.env` to `.gitignore` if it is not already there. Because this is a Vite variable prefixed with `VITE_`, the key is embedded in the client bundle; anyone who can load the built site can read it, so use a key with restricted quota and do not publish a build that contains a key you care about.

## Usage

Start the dev server:

```bash
npm run dev
```

Open the URL Vite prints (by default http://localhost:5173). Then:

1. Click **Start**. The first time, an intake form appears asking for a name, age, primary concern, goals, previous therapy and preferred approach. Fill it in and click **Begin Session**. The answers are stored in `localStorage` and can be reset later.
2. The app connects to Gemini Live, then asks for microphone permission. When the status dot turns green and the step label reads "Ready", speak. After about 1.2 seconds of silence the turn is sent and the model replies with audio; the text of the reply appears in the transcript.
3. Click **End** to close the session and release the microphone.

Other controls: **Import context** loads a `.txt` file (last 12,000 characters) as session memory; **Clear memory** wipes the stored transcript memory. Memory is kept in `localStorage` under `therapy_memory` and the last 4,000 characters are sent as system context on the next session.

Other scripts:

```bash
npm run build     # type-check and build to dist/
npm run preview   # serve the production build locally
npm run deploy    # publish dist/ to GitHub Pages with gh-pages
```

## How it works

- `src/App.tsx` holds nearly all the logic. On **Start** it creates a `GoogleGenAI` client with the key from `.env` and calls `ai.live.connect` with `responseModalities: [AUDIO]`, the `Puck` voice, and a system instruction that includes the intake profile and stored memory.
- After the session opens, it creates two `AudioContext`s (16 kHz input, 24 kHz output), requests the microphone, and uses a `ScriptProcessorNode` (4096-sample buffers) to convert float samples to 16-bit PCM, base64-encode them and send them with `sendRealtimeInput`.
- A simple RMS silence detector ends the user's turn after 1,200 ms below a threshold by sending `audioStreamEnd` and `turnComplete`.
- Incoming messages are decoded: audio parts are played through the output context; on `turnComplete` the text part is appended to the transcript and to the memory string.
- Connection and mic setup are wrapped in timeouts (15 s and 10 s). Any error or socket close tears down the processor, mic tracks, audio contexts and session.
- `src/components/Visualizer.tsx` and `src/utils/audio.ts` are helper files that are not currently imported by `App.tsx`.

## Project layout

```
nova-ai/
├── index.html                # Vite entry; loads Tailwind from CDN and Inter font
├── src/
│   ├── main.tsx              # React root
│   ├── App.tsx               # Session, audio pipeline, intake form, UI
│   ├── types.ts              # ConnectionStatus enum, Message type
│   ├── components/Visualizer.tsx   # Canvas frequency visualiser (unused)
│   └── utils/audio.ts        # PCM encode/decode helpers (unused)
├── public/
├── .env.example              # Variable names only; copy to .env locally
├── vite.config.ts
├── tsconfig*.json
└── eslint.config.js
```
