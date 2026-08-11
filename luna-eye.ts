/**
 * luna-eye.ts — 👁️ Luna Eye
 *
 * Gives the text-only model `deepseek-v4-flash` (provider `opencode-go`) vision
 * by using `gpt-5.6-luna` (provider `opencode-go`) as its eye, entirely inside
 * Pi's own model machinery (auth, providers, image serialization).
 *
 * What it does:
 *  1. Registers the `see` tool — the LLM can call it with an image path or
 *     base64 data to get a detailed textual description from gpt-5.6-luna.
 *  2. Intercepts the `input` event — when the user attaches images while the
 *     active model is text-only, they are described by the eye and the prompt
 *     is rewritten as text (the blind model never receives raw images).
 *  3. Safety net on the `context` event — any image parts that still reach the
 *     context (e.g. images returned by the `read` tool) are replaced with eye
 *     descriptions before the request is sent.
 *  4. `/eye` command — status and cache control.
 *
 * All model calls go through `ctx.modelRegistry.complete()` so the eye uses
 * Pi's own credential resolution for opencode-go — no hardcoded keys.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import { Type, type Static } from "typebox";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const EYE_PROVIDER = "opencode-go";
const EYE_MODEL = "gpt-5.6-luna";
const BRAIN_MODEL = "deepseek-v4-flash";
const EYE_TIMEOUT_MS = 180_000;
const MAX_CACHE_ENTRIES = 96;

/** Description cache: image hash -> eye description. Avoids re-paying for the same image. */
const descriptionCache = new Map<string, string>();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isBlind(model: { input: readonly string[] } | undefined): boolean {
  // No active model -> assume blind (safe default: translate images).
  return !model || !model.input.includes("image");
}

function imageKey(img: ImageContent, instruction: string): string {
  return createHash("sha256")
    .update(`${img.mimeType}\n${img.data}\n${instruction}`)
    .digest("hex")
    .slice(0, 24);
}

function buildInstruction(userInstruction?: string): string {
  const base =
    "You are the Luna Eye — a meticulous visual perception module. A text-only coding agent (deepseek-v4-flash) is using you as its eyes: it cannot see images, so your description is the only information it receives. Be precise, structured, and exhaustive.";
  const task = userInstruction?.trim()
    ? `The agent asked you to focus on this:\n${userInstruction.trim()}`
    : "Describe everything in the image in meticulous detail: main subjects, layout, colors, any visible text (quote it verbatim — code, error messages, labels, UI strings), UI elements, and anything else notable. For screenshots, cover the entire screen.";
  return `${base}\n\n${task}`;
}

function describeError(err: unknown): string {
  if (err instanceof DOMException && err.name === "TimeoutError") {
    return `Luna Eye timed out after ${EYE_TIMEOUT_MS / 1000}s`;
  }
  return err instanceof Error ? err.message : String(err);
}

/**
 * Send one image to gpt-5.6-luna through Pi's own model registry and return
 * the textual description. Cached per image hash.
 */
async function describeImage(
  img: ImageContent,
  instruction: string,
  signal: AbortSignal | undefined,
  ctx: ExtensionContext,
): Promise<string> {
  const key = imageKey(img, instruction);
  const cached = descriptionCache.get(key);
  if (cached !== undefined) return cached;

  const model = ctx.modelRegistry.find(EYE_PROVIDER, EYE_MODEL);
  if (!model) {
    throw new Error(`Eye model ${EYE_PROVIDER}/${EYE_MODEL} is not registered`);
  }

  const timeout = AbortSignal.timeout(EYE_TIMEOUT_MS);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;

  const result = await ctx.modelRegistry.complete(
    model,
    {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: instruction },
            { type: "image", data: img.data, mimeType: img.mimeType },
          ],
          timestamp: Date.now(),
        },
      ],
    },
    { signal: combined, reasoningEffort: "low", maxTokens: 8192 },
  );

  const text = result.content
    .filter((c): c is TextContent => c.type === "text")
    .map((c) => c.text)
    .join("\n")
    .trim();
  if (!text) {
    throw new Error("Luna Eye returned an empty description");
  }

  if (descriptionCache.size >= MAX_CACHE_ENTRIES) {
    const oldest = descriptionCache.keys().next().value;
    if (oldest !== undefined) descriptionCache.delete(oldest);
  }
  descriptionCache.set(key, text);
  return text;
}

/** Describe several images, one eye call each, in parallel. */
function describeImages(
  images: ImageContent[],
  instruction: string,
  signal: AbortSignal | undefined,
  ctx: ExtensionContext,
): Promise<string[]> {
  return Promise.all(images.map((img) => describeImage(img, instruction, signal, ctx)));
}

const MIME_BY_EXT: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  bmp: "image/bmp",
  svg: "image/svg+xml",
  avif: "image/avif",
  heic: "image/heic",
  ico: "image/x-icon",
  tiff: "image/tiff",
  tif: "image/tiff",
};

function sniffMime(buf: Buffer, path?: string): string {
  if (buf.length >= 8 && buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return "image/png";
  if (buf.length >= 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return "image/jpeg";
  if (buf.length >= 6 && buf.toString("ascii", 0, 4) === "GIF8") return "image/gif";
  if (buf.length >= 12 && buf.toString("ascii", 0, 4) === "RIFF" && buf.toString("ascii", 8, 12) === "WEBP") return "image/webp";
  if (buf.length >= 2 && buf[0] === 0x42 && buf[1] === 0x4d) return "image/bmp";
  const head = buf.subarray(0, 128).toString("latin1");
  if (head.includes("<svg")) return "image/svg+xml";
  const ext = (path ?? "").toLowerCase().split(".").pop() ?? "";
  return MIME_BY_EXT[ext] ?? "image/png";
}

/** Accepts a full data URL or raw base64. */
function parseDataParam(data: string): { data: string; mimeType: string } {
  const cleaned = data.replace(/\s+/g, "");
  const m = /^data:([^;,]+)(;base64)?,(.*)$/s.exec(cleaned);
  if (m) return { data: m[3], mimeType: m[1] || "image/png" };
  return { data: cleaned, mimeType: "image/png" };
}

// ---------------------------------------------------------------------------
// `see` tool
// ---------------------------------------------------------------------------

const SeeParams = Type.Object({
  path: Type.Optional(
    Type.String({ description: "Path to the image file (absolute or relative to the project)" }),
  ),
  data: Type.Optional(
    Type.String({
      description: "Image data as a base64 data URL (data:image/png;base64,...) or raw base64",
    }),
  ),
  instruction: Type.Optional(
    Type.String({
      description:
        "What to look for (e.g. 'What error message is shown?'). Defaults to a full detailed description",
    }),
  ),
});
type SeeParams = Static<typeof SeeParams>;

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function lunaEye(pi: ExtensionAPI) {
  pi.registerTool({
    name: "see",
    label: "👁️ See (Luna Eye)",
    description:
      "Look at an image using gpt-5.6-luna (your eye model on opencode-go) and return a detailed textual description. This is the ONLY way you can perceive visual content — use it whenever you need to see an image file, screenshot, diagram, UI mockup, or attached picture. Pass either `path` (image file) or `data` (base64 data URL / raw base64).",
    promptSnippet: "👁️ see(path|data, instruction?) — describe an image via gpt-5.6-luna (your eye)",
    promptGuidelines: [
      "You cannot process images directly — you are a text-only model. Whenever the user asks about an image, screenshot, diagram, or anything visual, call the `see` tool instead of guessing.",
      "If you used the `read` tool on an image file, call `see` with that same path to actually perceive what it shows.",
    ],
    parameters: SeeParams,
    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: `👁️ Luna Eye (${EYE_MODEL}) is looking…` }], details: {} });
      try {
        let img: ImageContent;
        if (params.path) {
          const abs = resolve(ctx.cwd, params.path);
          const buf = await readFile(abs);
          img = {
            type: "image",
            data: buf.toString("base64"),
            mimeType: sniffMime(buf, abs),
          };
        } else if (params.data) {
          const parsed = parseDataParam(params.data);
          img = { type: "image", data: parsed.data, mimeType: parsed.mimeType };
        } else {
          throw new Error("Provide either `path` (image file) or `data` (base64 image)");
        }

        const description = await describeImage(
          img,
          buildInstruction(params.instruction),
          signal,
          ctx,
        );
        return {
          content: [
            {
              type: "text",
              text: `[👁️ Luna Eye (${EYE_MODEL}) saw the image${params.path ? ` at ${params.path}` : ""}:\n${description}\n]`,
            },
          ],
          details: { eye: EYE_MODEL, provider: EYE_PROVIDER, source: params.path ?? "data" },
        };
      } catch (err) {
        return {
          content: [{ type: "text", text: `👁️ Luna Eye failed: ${describeError(err)}` }],
          isError: true,
          details: { eye: EYE_MODEL, provider: EYE_PROVIDER, error: describeError(err) },
        };
      }
    },
  });

  // -------------------------------------------------------------------------
  // Translate user-attached images to text when the active model is blind
  // -------------------------------------------------------------------------
  pi.on("input", async (event, ctx) => {
    if (!event.images || event.images.length === 0) return;
    if (!isBlind(ctx.model)) return; // vision model: leave images untouched

    const label =
      event.images.length === 1
        ? "an attached image"
        : `${event.images.length} attached images`;
    try {
      const descriptions = await describeImages(
        event.images,
        "The user attached this image to their message. Describe it thoroughly so the text-only agent can answer correctly.",
        ctx.signal,
        ctx,
      );
      const blocks = descriptions
        .map((d, i) => `Attached image #${i + 1}:\n${d}`)
        .join("\n\n");
      return {
        action: "transform",
        text: `${event.text}\n\n[👁️ Luna Eye (${EYE_MODEL}) — ${label} described for the text-only model (${BRAIN_MODEL}):\n${blocks}\n]`,
      };
    } catch (err) {
      ctx.ui.notify(`Luna Eye failed to describe the attached image: ${describeError(err)}`, "error");
      // Drop the images anyway so the blind model's request does not break.
      return {
        action: "transform",
        text: `${event.text}\n\n[👁️ Luna Eye failed to describe ${label}: ${describeError(err)}]`,
      };
    }
  });

  // -------------------------------------------------------------------------
  // Safety net: replace image parts in context for blind models
  // (covers images returned by tools such as `read`, and images injected by
  // other extensions). Cached per image, so repeats are free.
  // -------------------------------------------------------------------------
  pi.on("context", async (event, ctx) => {
    if (!isBlind(ctx.model)) return;
    if (event.messages.length === 0) return;

    let changed = false;
    const messages: AgentMessage[] = [];
    for (const msg of event.messages) {
      if (msg.role !== "user" && msg.role !== "toolResult") {
        messages.push(msg);
        continue;
      }
      if (typeof msg.content === "string") {
        messages.push(msg);
        continue;
      }
      const images = msg.content.filter((c) => c.type === "image");
      if (images.length === 0) {
        messages.push(msg);
        continue;
      }
      changed = true;
      const newContent = [...msg.content];
      // Replace each image part with a placeholder text part, then fill in the
      // descriptions (one eye call per unique image, in parallel).
      for (let i = 0; i < newContent.length; i++) {
        if (newContent[i].type === "image") {
          newContent[i] = { type: "text", text: "[👁️ Luna Eye image placeholder]" };
        }
      }
      messages.push({ ...msg, content: newContent });
    }
    if (!changed) return;

    // Describe all images from all affected messages, in parallel.
    const allImages: ImageContent[] = [];
    for (const msg of messages) {
      if ((msg.role === "user" || msg.role === "toolResult") && Array.isArray(msg.content)) {
        for (const part of msg.content) {
          if (part.type === "image") allImages.push(part);
        }
      }
    }
    try {
      const descriptions = await describeImages(
        allImages,
        "This image is part of the conversation context. Describe it in detail so the text-only agent can work with it.",
        ctx.signal,
        ctx,
      );
      let idx = 0;
      for (const msg of messages) {
        if ((msg.role === "user" || msg.role === "toolResult") && Array.isArray(msg.content)) {
          for (let i = 0; i < msg.content.length; i++) {
            const part = msg.content[i];
            if (part.type === "text" && part.text === "[👁️ Luna Eye image placeholder]") {
              msg.content[i] = { type: "text", text: `[👁️ Luna Eye (${EYE_MODEL}) saw an image:\n${descriptions[idx++]}\n]` };
            }
          }
        }
      }
      return { messages };
    } catch (err) {
      // Never fail the turn because the eye failed: keep textual placeholders.
      for (const msg of messages) {
        if ((msg.role === "user" || msg.role === "toolResult") && Array.isArray(msg.content)) {
          for (let i = 0; i < msg.content.length; i++) {
            const part = msg.content[i];
            if (part.type === "text" && part.text === "[👁️ Luna Eye image placeholder]") {
              msg.content[i] = { type: "text", text: `[👁️ Luna Eye failed to describe an image: ${describeError(err)}]` };
            }
          }
        }
      }
      return { messages };
    }
  });

  // -------------------------------------------------------------------------
  // Notifications
  // -------------------------------------------------------------------------
  pi.on("session_start", async (_event, ctx) => {
    if (ctx.hasUI && isBlind(ctx.model)) {
      ctx.ui.notify(
        `👁️ Luna Eye active — ${EYE_PROVIDER}/${EYE_MODEL} sees for ${ctx.model?.id ?? "the current model"}`,
        "info",
      );
    }
  });

  pi.on("model_select", async (event, ctx) => {
    if (ctx.hasUI && event.model && !event.model.input.includes("image")) {
      ctx.ui.notify(
        `👁️ ${event.model.id} is text-only — Luna Eye (${EYE_MODEL}) is available via the see tool`,
        "info",
      );
    }
  });

  // -------------------------------------------------------------------------
  // /eye command
  // -------------------------------------------------------------------------
  pi.registerCommand("eye", {
    description:
      "Luna Eye status and cache control. Usage: /eye [status|clear] — status shows the eye model and whether the active model is blind; clear empties the description cache.",
    handler: async (args, ctx) => {
      const arg = (args ?? "").trim().toLowerCase();
      if (arg === "clear") {
        const cleared = descriptionCache.size;
        descriptionCache.clear();
        ctx.ui.notify(`👁️ Luna Eye: cleared ${cleared} cached description(s)`, "info");
        return;
      }
      const model = ctx.model;
      const eye = ctx.modelRegistry.find(EYE_PROVIDER, EYE_MODEL);
      const lines = [
        `Eye model : ${EYE_PROVIDER}/${EYE_MODEL} ${eye ? (eye.input.includes("image") ? "(vision ✓)" : "(no vision!)") : "(not registered)"}`,
        `Active    : ${model ? `${model.provider}/${model.id}` : "none"} — ${isBlind(model) ? `text-only (Luna Eye translates images for ${BRAIN_MODEL})` : "has vision (Luna Eye passive)"}`,
        `Cache     : ${descriptionCache.size} described image(s)`,
        `Usage     : call the see tool, or attach an image to a message`,
      ];
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });
}
