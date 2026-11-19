import test from "node:test"
import assert from "node:assert/strict"

process.env.GITHUB_TOKEN = "test-token"
process.env.GITHUB_REPOSITORY = "garnet-labs/litellm"
process.env.PR_NUMBER = "1"
process.env.HEAD_SHA = "a".repeat(40)
const { main } = await import("./garnet-evidence-mirror.mjs")

test("body marker opts PR out of comment reads and full-description writes", async () => {
  const calls = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init })
    return { ok: true, json: async () => ({ head: { sha: process.env.HEAD_SHA }, body: "Human description\n<!-- garnet:record-summary:opt-in -->" }) }
  }
  await main()
  assert.equal(calls.length, 1)
  assert.equal(calls[0].init.method, undefined)
})

test("old label without body marker retains missing-evidence mirroring", async () => {
  const calls = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init })
    return { ok: true, json: async () => url.includes("/comments?") ? [] : { head: { sha: process.env.HEAD_SHA }, labels: [{ name: "garnet-record-summary" }], body: "Human description" } }
  }
  await main()
  const write = calls.find(({ init }) => init.method === "PATCH")
  assert.ok(write)
  assert.match(JSON.parse(write.init.body).body, /Human description[\s\S]*No runtime evidence/)
})

test("null PR body retains missing-evidence mirroring", async () => {
  const calls = []
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init })
    return { ok: true, json: async () => url.includes("/comments?") ? [] : { head: { sha: process.env.HEAD_SHA }, body: null } }
  }
  await main()
  assert.match(JSON.parse(calls.find(({ init }) => init.method === "PATCH").init.body).body, /No runtime evidence/)
})

test("stale heads do not read comments or write", async () => {
  let calls = 0
  globalThis.fetch = async () => {
    calls++
    return { ok: true, json: async () => ({ head: { sha: "b".repeat(40) }, labels: [] }) }
  }
  await main()
  assert.equal(calls, 1)
})

test("PR API errors fail without a write", async () => {
  let calls = 0
  globalThis.fetch = async () => {
    calls++
    return { ok: false, status: 403, text: async () => "denied" }
  }
  await assert.rejects(main(), /403 denied/)
  assert.equal(calls, 1)
})
