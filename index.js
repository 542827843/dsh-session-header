/**
 * Inject one HTTP header (default `x-session-id`) onto every LLM provider
 * request the harness sends, carrying the harness session id of that exact
 * call — so turns, compaction/title helper calls, and in-process subagent
 * children each report their own session.
 *
 * The harness has no per-request header seam: `GenerateOptions` carries no
 * headers field, and each adapter builds its own wire headers inside
 * `stream()`. The two official interception points compose into one here:
 *
 * - the `llm/stream` waterfall names the calls that are LLM calls and carries
 *   `options.sessionId`, and
 * - a `globalThis.fetch` patch adds the header, so every fetch-based adapter
 *   (llm-deepseek, llm-pi-ai, and any SDK whose transport bottoms out in
 *   global fetch) is covered without touching adapter code.
 *
 * Context propagation uses AsyncLocalStorage: each `iterator.next()` resumes
 * the adapter's stream inside `als.run()`, so the adapter's internal `fetch`
 * lands in the store, while unrelated fetches (web RPC, telemetry, tools)
 * see no store and pass through untouched.
 *
 * A fixed value can be configured instead of the live session id (sent
 * verbatim); the live id's `session-` branding prefix is stripped so a plain
 * UUID goes on the wire. Calls with neither a configured value nor a session
 * id get no header. A header anyone else already set is never overwritten
 * (`Headers` matching is case-insensitive, the same rule the wire uses).
 * Unloading the plugin restores the original fetch.
 *
 * @module dsh-session-header
 */

import { AsyncLocalStorage } from 'node:async_hooks'
import Schema from '@deepseek-ai/schemastery'

export const name = 'dsh-session-header'

// Hard dependency: nothing to do until the llm runtime exists.
export const inject = ['llm']

export const Config = Schema.object({
  /** Header name to inject; HTTP field names are case-insensitive on the wire. */
  header: Schema.string().default('x-session-id'),
  /**
   * Fixed header value. Unset means "use the harness session id of the call
   * in flight" (`GenerateOptions.sessionId`); calls with neither get no header.
   */
  value: Schema.string(),
})

/**
 * Plugin entry. `config.header` names the header, `config.value` optionally
 * fixes its value; per call, the AsyncLocalStorage store carries both plus the
 * resolved value, and a present store marks "this is an LLM fetch".
 */
export function apply(ctx, config) {
  const als = new AsyncLocalStorage()
  const originalFetch = globalThis.fetch

  const patchedFetch = (input, init) => {
    const injection = als.getStore()
    if (injection === undefined || injection.value === undefined) {
      return originalFetch(input, init)
    }
    // Collect headers from both fetch() spellings: a Request object carries
    // its own, and init.headers overrides them per the fetch standard.
    const headers = new Headers(input instanceof Request ? input.headers : undefined)
    if (init?.headers !== undefined) {
      for (const [key, value] of new Headers(init.headers)) headers.set(key, value)
    }
    // Inject only when absent; never overwrite a value someone else set.
    if (headers.has(injection.header)) {
      return originalFetch(input, init)
    }
    headers.set(injection.header, injection.value)
    // A bare Request input owns its headers; rebuild it so the original —
    // which may be reused by the caller — keeps arriving providers without us.
    if (input instanceof Request && init === undefined) {
      return originalFetch(new Request(input, { headers }))
    }
    return originalFetch(input, { ...init, headers })
  }

  globalThis.fetch = patchedFetch
  // cordis `ctx.effect`: the callback runs IMMEDIATELY (setup); its RETURN
  // VALUE is the disposer collected for fiber unload. v0.1.0 ran the restore
  // in the callback body — undoing the patch in the same tick — so the fix
  // returns the restore as the disposer.
  ctx.effect(() => () => {
    if (globalThis.fetch === patchedFetch) {
      globalThis.fetch = originalFetch
    } else {
      // Someone else replaced fetch after we loaded; their patch fronts ours,
      // so restoring ours would silently drop theirs.
      ctx.logger.warn('dsh-session-header: global fetch was replaced after load; not restoring')
    }
  })

  ctx.on('llm/stream', async function* (options, next) {
    const inner = next()
    const iterator = inner[Symbol.asyncIterator]()
    const scope = {
      header: config.header,
      value:
        config.value ??
        (options.sessionId !== undefined ? String(options.sessionId).replace(/^session-/, '') : undefined),
    }
    let exhausted = false
    try {
      while (true) {
        // Each adapter step runs inside the store, so its internal fetch
        // (whenever in the call it happens) sees this call's injection facts.
        const result = await als.run(scope, () => iterator.next())
        if (result.done) {
          exhausted = true
          return
        }
        yield result.value
      }
    } finally {
      if (!exhausted) await iterator.return?.()
    }
  })
}
