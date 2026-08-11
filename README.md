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

Descriptions are cached per image + instruction (sha256), so repeated turns never re-pay for
the same query. `/eye clear` empties the cache.

The eye model is runtime-switchable and persisted to `~/.pi/agent/luna-eye.json`.

## Commands

- `/eye` — **paged interactive picker with live search** (like `/model`):
  full-screen, keyboard-navigable list of all vision-capable models across
  your configured providers. Just start **typing to filter** (fuzzy, best match
  first, e.g. `mimo`, `claude`, `qwen3`), **max 6 visible rows** with a
  `(n/total)` page indicator as you scroll (↑/↓, wrap-around). Current eye
  listed first with `✓`. Enter selects the highlighted model (or the best
  match from the search box), Esc cancels. In RPC mode it becomes a plain
  select dialog; in print mode a numbered list.
- `/eye set <n>` — pick by list number (scriptable, works in `-p` mode)
- `/eye set <model>` — e.g. `/eye set mimo-v2.5` (if unique)
- `/eye set <provider>/<model>` — e.g. `/eye set opencode-go/mimo-v2.5`
- `/eye clear` — clear the description cache

Switching the eye model also clears the cache, so stale descriptions from the
previous model are never reused.

## Configuration

Defaults at the top of `luna-eye.ts`:

```ts
const EYE_PROVIDER = "opencode-go";
const EYE_MODEL = "gpt-5.6-luna";
const BRAIN_MODEL = "deepseek-v4-flash";
const EYE_TIMEOUT_MS = 180_000;
const MAX_CACHE_ENTRIES = 96;
```

The effective eye model is overridden at runtime by `/eye set`, which persists
to `~/.pi/agent/luna-eye.json`:

```json
{ "eyeProvider": "opencode-go", "eyeModel": "mimo-v2.5" }
```

Delete that file to fall back to the code defaults.

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

Eye switched to mimo-v2.5 via /eye set, then:

pi -p "Use the see tool to look at /tmp/eye-mimo1.png and tell me exactly what text is in the image."
→ "LUNA EYE TEST 2026 / The password is: zebra-42"   (mimo-v2.5 verbatim text ✓)

pi -p "Call the see tool TWICE on /tmp/eye-mimo2.png. First: 'What color is the circle?' Second: 'What color is the background?'"
→ "blue" / "bright, saturated red"                       (mimo-v2.5 instructions ✓)
```
