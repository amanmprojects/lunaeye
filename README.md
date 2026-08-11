# 👁️ Luna Eye

A Pi extension that gives the **text-only** model `deepseek-v4-flash` vision by
using **`gpt-5.6-luna`** as its eye. Both models run on the `opencode-go`
provider, and the eye is called through Pi's own model registry — so it uses
Pi's credential resolution (no hardcoded keys).

## Installation

The extension is a single file: [`luna-eye.ts`](luna-eye.ts).

Install globally (all projects):

```bash
cp luna-eye.ts ~/.pi/agent/extensions/luna-eye.ts
```

or symlink it so the installed extension tracks this repo:

```bash
ln -s "$(pwd)/luna-eye.ts" ~/.pi/agent/extensions/luna-eye.ts
```

Reload with `/reload` or restart Pi. (If you previously installed the
project-local copy at `.pi/extensions/luna-eye.ts`, remove it to avoid double
registration.)

## How it works

| Path | What happens |
|------|--------------|
| **`see` tool** | deepseek-v4-flash can call `see(path \| data, instruction?)` to look at any image file or base64 data. The tool sends it to gpt-5.6-luna and returns a detailed textual description. |
| **Attached images** | When you attach an image to a message (drag & drop, or `pi @image.png "…"`), the `input` event intercepts it, the eye describes it, and the prompt is rewritten as text — the blind model never receives raw pixels. |
| **Context safety net** | If image parts still reach the context by any other route (e.g. images returned by tools), the `context` event replaces them with eye descriptions before the request is sent. |

Descriptions are cached per-image (sha256), so repeated turns never re-pay for
the same image. `/eye clear` empties the cache.

## Commands

- `/eye` — show eye model, active model vision status, cache size
- `/eye status` — same as above
- `/eye clear` — clear the description cache

## Configuration

Constants at the top of `luna-eye.ts`:

```ts
const EYE_PROVIDER = "opencode-go";
const EYE_MODEL = "gpt-5.6-luna";
const BRAIN_MODEL = "deepseek-v4-flash";
const EYE_TIMEOUT_MS = 180_000;
const MAX_CACHE_ENTRIES = 96;
```

The eye call uses `reasoningEffort: "low"` for fast, cheap perception. When the
active model has native vision (e.g. gpt-5.6-luna itself), Luna Eye goes fully
passive and images pass through untouched.

## Verification

All paths were smoke-tested in Pi:

```
pi -p "Use the see tool to look at /tmp/eye-test.png and tell me exactly what text is in the image."
→ "LUNA EYE TEST 2026 / The password is: zebra-42"   (see tool ✓)

pi -p "@/tmp/eye-test2.png" "This image was attached. Describe what colors it contains."
→ blue→red vertical gradient, colors listed             (input transform ✓)

pi -p "Use the read tool on /tmp/eye-test.png, then tell me what the image shows."
→ model detected blindness and called `see`            (tool fallback ✓)
```
