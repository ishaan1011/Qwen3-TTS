// Streaming TTS chat client.
// Receives:
//   - JSON text frames: {type: "llm_token"|"phrase"|"audio_start"|"audio_end"|"error"|"reset_ack", ...}
//   - Binary frames: 24 kHz mono s16le PCM chunks
// Plays incoming PCM gaplessly via Web Audio scheduling.

const SAMPLE_RATE = 24000;

const $ = (id) => document.getElementById(id);
const messages = $("messages");
const input = $("input");
const form = $("form");
const sendBtn = $("send");
const resetBtn = $("reset");
const statusEl = $("status");

let ws;
let audioCtx;
let nextStartTime = 0;          // wallclock (audioCtx.currentTime) of next scheduled buffer
let currentAssistantEl = null;  // DOM element of the assistant bubble being filled
let inFlight = false;

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

function ensureAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function appendMessage(role, text = "") {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function appendPhraseTag(el, text) {
  let phrases = el.querySelector(".phrases");
  if (!phrases) {
    phrases = document.createElement("div");
    phrases.className = "phrases";
    el.appendChild(phrases);
  }
  phrases.textContent += (phrases.textContent ? " · " : "") + text;
}

// Convert Int16 PCM -> Float32, schedule via Web Audio for gapless playback.
function playPcm(arrayBuffer) {
  const ctx = ensureAudioContext();
  // Browsers can auto-suspend AudioContext after long silence; resume
  // defensively before scheduling each chunk so we don't lose audio after
  // a long synth wait.
  if (ctx.state === "suspended") ctx.resume();
  const pcm = new Int16Array(arrayBuffer);
  if (pcm.length === 0) return;
  const buf = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
  const channel = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const start = Math.max(ctx.currentTime + 0.02, nextStartTime);
  src.start(start);
  nextStartTime = start + buf.duration;
  if (window._dbg) console.log("playPcm", pcm.length, "samples", "ctx.state", ctx.state, "start", start.toFixed(2), "next", nextStartTime.toFixed(2));
}

// Set window._dbg = true in the browser console to see playPcm logs.
window._dbg = false;

function connect() {
  const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => setStatus("connected", "ok");
  ws.onclose = () => {
    setStatus("disconnected — refresh to reconnect", "err");
    sendBtn.disabled = true;
  };
  ws.onerror = () => setStatus("error", "err");

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }
      switch (msg.type) {
        case "audio_start":
          currentAssistantEl = appendMessage("assistant", "");
          nextStartTime = 0; // reset playback clock per utterance
          break;
        case "llm_token":
          if (currentAssistantEl) {
            // append text before any .phrases child
            const phrases = currentAssistantEl.querySelector(".phrases");
            if (phrases) {
              currentAssistantEl.insertBefore(document.createTextNode(msg.delta), phrases);
            } else {
              currentAssistantEl.appendChild(document.createTextNode(msg.delta));
            }
            messages.scrollTop = messages.scrollHeight;
          }
          break;
        case "phrase":
          if (currentAssistantEl) appendPhraseTag(currentAssistantEl, msg.text);
          break;
        case "audio_end":
          inFlight = false;
          sendBtn.disabled = false;
          currentAssistantEl = null;
          input.focus();
          break;
        case "error":
          setStatus("error: " + msg.message, "err");
          inFlight = false;
          sendBtn.disabled = false;
          break;
        case "reset_ack":
          messages.innerHTML = "";
          setStatus("reset", "ok");
          break;
      }
    } else {
      playPcm(event.data);
    }
  };
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || inFlight || ws.readyState !== WebSocket.OPEN) return;
  ensureAudioContext(); // user-gesture init
  appendMessage("user", text);
  ws.send(JSON.stringify({ type: "user_message", content: text }));
  input.value = "";
  inFlight = true;
  sendBtn.disabled = true;
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

resetBtn.addEventListener("click", () => {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "reset" }));
});

connect();
