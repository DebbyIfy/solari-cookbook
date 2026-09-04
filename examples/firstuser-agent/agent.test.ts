/**
 * Deterministic fixture tests for detectBlockingOverlay() — the pure
 * geometry decision behind FirstUser's blocking-overlay awareness.
 *
 * These are plain-data fixtures (no browser, no DOM). The actual
 * fixed/absolute-positioned candidate list is collected browser-side in
 * observe() (see agent.ts); this only tests the decision made from that
 * data, which is the part that can meaningfully regress.
 *
 * Run with: npx tsx --test agent.test.ts
 */

import "dotenv/config"
import { test } from "node:test"
import assert from "node:assert/strict"
import {
  detectBlockingOverlay,
  findOverlayActions,
  computeClickableSamplePoints,
  CLICKABLE_SAMPLE_MIN_HITS,
  computeOverlaySamplePoints,
  OVERLAY_TOPMOST_SAMPLE_MIN_HITS,
  type OverlayCandidate,
  type Rect,
} from "./agent.ts"

const VIEWPORT = { width: 1280, height: 800 }

// topmostHits defaults to fully topmost (3 of 3) — every existing fixture
// below is meant to represent something genuinely painted on top of the
// page, so only tests that specifically exercise the "large but not
// actually on top" case need to override it.
function candidate(overrides: Partial<OverlayCandidate>): OverlayCandidate {
  return {
    position: "fixed",
    display: "block",
    visibility: "visible",
    opacity: 1,
    rect: { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 },
    text: "",
    topmostHits: 3,
    ...overrides,
  }
}

test("blocking cookie modal: full-screen backdrop is detected as present", () => {
  const cookieModal = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    text: "We use cookies to improve your experience. Accept all / Reject all",
  })

  const result = detectBlockingOverlay([cookieModal], VIEWPORT)

  assert.equal(result.present, true)
  assert.ok(result.coverageRatio! >= 0.15)
  assert.match(result.text!, /cookies/i)
})

test("non-blocking chat widget: small corner bubble is NOT detected as blocking", () => {
  const chatBubble = candidate({
    rect: { left: 1200, top: 720, right: 1260, bottom: 780, width: 60, height: 60 },
    text: "Chat with us",
  })

  const result = detectBlockingOverlay([chatBubble], VIEWPORT)

  assert.equal(result.present, false)
})

test("login wall: full-screen gated overlay is detected as present", () => {
  const loginWall = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    text: "Sign in to continue reading this page. Email Password Log in",
  })

  const result = detectBlockingOverlay([loginWall], VIEWPORT)

  assert.equal(result.present, true)
  assert.match(result.text!, /sign in|log in/i)
})

test("hidden or transparent large elements are ignored even if geometrically full-screen", () => {
  const hiddenModal = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    display: "none",
    text: "A modal that is not actually shown",
  })
  const invisibleModal = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    visibility: "hidden",
    text: "A modal that is not actually shown",
  })
  const transparentModal = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    opacity: 0,
    text: "A modal that is not actually shown",
  })

  assert.equal(detectBlockingOverlay([hiddenModal], VIEWPORT).present, false)
  assert.equal(detectBlockingOverlay([invisibleModal], VIEWPORT).present, false)
  assert.equal(detectBlockingOverlay([transparentModal], VIEWPORT).present, false)
})

// -----------------------------------
// Regression: geometry alone is not evidence. A large `position: fixed`/
// `absolute` element that is visible, opaque, and covers most of the
// viewport can still be purely decorative if it sits behind the page's
// real content in paint order — e.g. a hero section's illustration
// wrapper. This is exactly what caused FirstUser to report a "blocking
// overlay" on a page that had none.
// -----------------------------------

test("large decorative background element behind real content is NOT detected as blocking", () => {
  const decorativeHeroWrapper = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    text: "Failed payments in Naira aren't gone. They're stuck. See what it costs How it works",
    // Passes every existing geometry/visibility check, but elementFromPoint()
    // never reports it (or a descendant of it) as topmost — nothing about
    // it was actually painted over the visitor's view.
    topmostHits: 0,
  })

  const result = detectBlockingOverlay([decorativeHeroWrapper], VIEWPORT)

  assert.equal(result.present, false)
})

test("below-majority topmost hits are not enough to count as blocking", () => {
  const mostlyBackgroundElement = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    text: "Large element with only one incidental topmost sample",
    topmostHits: OVERLAY_TOPMOST_SAMPLE_MIN_HITS - 1,
  })

  const result = detectBlockingOverlay([mostlyBackgroundElement], VIEWPORT)

  assert.equal(result.present, false)
})

test("a genuine overlay confirmed topmost at a majority of sample points is still detected", () => {
  const realModal = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    text: "Sign in to continue. Email Password Log in",
    topmostHits: OVERLAY_TOPMOST_SAMPLE_MIN_HITS,
  })

  const result = detectBlockingOverlay([realModal], VIEWPORT)

  assert.equal(result.present, true)
  assert.match(result.text!, /sign in/i)
})

test("no candidates: reports not present", () => {
  const result = detectBlockingOverlay([], VIEWPORT)
  assert.equal(result.present, false)
})

test("multiple candidates: picks the one with the highest coverage", () => {
  const smallBanner = candidate({
    rect: { left: 0, top: 700, right: 1280, bottom: 800, width: 1280, height: 100 },
    text: "Small non-material banner",
  })
  const bigModal = candidate({
    rect: { left: 140, top: 100, right: 1140, bottom: 700, width: 1000, height: 600 },
    text: "The actual blocking modal",
  })

  const result = detectBlockingOverlay([smallBanner, bigModal], VIEWPORT)

  assert.equal(result.present, true)
  assert.match(result.text!, /actual blocking modal/)
})

test("degenerate zero-area viewport does not throw and reports not present", () => {
  const result = detectBlockingOverlay(
    [candidate({ rect: { left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 } })],
    { width: 0, height: 0 }
  )
  assert.equal(result.present, false)
})

test("detected overlay carries the winning candidate's rect", () => {
  const cookieModal = candidate({
    rect: { left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 },
    text: "Cookie banner",
  })

  const result = detectBlockingOverlay([cookieModal], VIEWPORT)

  assert.equal(result.present, true)
  assert.deepEqual(result.rect, cookieModal.rect)
})

// -----------------------------------
// findOverlayActions() — which page buttons/links belong to the overlay
// -----------------------------------

function rect(overrides: Partial<Rect>): Rect {
  return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0, ...overrides }
}

test("overlay actions: buttons overlapping the overlay rect are surfaced with their real index", () => {
  const overlay = detectBlockingOverlay(
    [candidate({ rect: { left: 240, top: 200, right: 1040, bottom: 600, width: 800, height: 400 }, text: "Cookie banner" })],
    VIEWPORT
  )

  const buttons = [
    { index: 0, text: "Reject all", rect: rect({ left: 300, top: 500, right: 420, bottom: 540, width: 120, height: 40 }) },
    { index: 1, text: "Accept all", rect: rect({ left: 440, top: 500, right: 560, bottom: 540, width: 120, height: 40 }) },
    { index: 2, text: "Pricing", rect: rect({ left: 20, top: 20, right: 90, bottom: 50, width: 70, height: 30 }) }, // outside overlay
  ]
  const links: Array<{ index: number; text: string; rect: Rect }> = []

  const actions = findOverlayActions(overlay, buttons, links)

  assert.deepEqual(actions, [
    { type: "button", text: "Reject all", index: 0 },
    { type: "button", text: "Accept all", index: 1 },
  ])
})

test("overlay actions: empty when no overlay is present", () => {
  const noOverlay = detectBlockingOverlay([], VIEWPORT)
  const buttons = [{ index: 0, text: "Reject all", rect: rect({ left: 0, top: 0, right: 100, bottom: 40, width: 100, height: 40 }) }]

  assert.deepEqual(findOverlayActions(noOverlay, buttons, []), [])
})

test("overlay actions: links outside the overlay's rect are excluded", () => {
  const overlay = detectBlockingOverlay(
    [candidate({ rect: { left: 0, top: 0, right: 1280, bottom: 200, width: 1280, height: 200 }, text: "Top banner" })],
    VIEWPORT
  )

  const links = [
    { index: 0, text: "Manage preferences", rect: rect({ left: 10, top: 10, right: 150, bottom: 50, width: 140, height: 40 }) },
    { index: 1, text: "Product roadmap", rect: rect({ left: 10, top: 700, right: 150, bottom: 740, width: 140, height: 40 }) },
  ]

  const actions = findOverlayActions(overlay, [], links)

  assert.deepEqual(actions, [{ type: "link", text: "Manage preferences", index: 0 }])
})

// -----------------------------------
// computeClickableSamplePoints() — the geometry behind isTargetClickable()'s
// hardened, multi-point occlusion check (added after a real run against
// Notion showed a single center-point check reporting a fully on-screen,
// visually unobstructed button as occluded).
// -----------------------------------

test("clickable sample points: returns 5 points (center + 4 inset corners), all inside the rect", () => {
  const buttonRect = rect({ left: 100, top: 200, right: 300, bottom: 240, width: 200, height: 40 })

  const points = computeClickableSamplePoints(buttonRect)

  assert.equal(points.length, 5)
  for (const point of points) {
    assert.ok(point.x >= buttonRect.left && point.x <= buttonRect.right, `x=${point.x} outside rect`)
    assert.ok(point.y >= buttonRect.top && point.y <= buttonRect.bottom, `y=${point.y} outside rect`)
  }
})

test("clickable sample points: center point is the exact geometric center", () => {
  const buttonRect = rect({ left: 0, top: 0, right: 100, bottom: 50, width: 100, height: 50 })

  const [center] = computeClickableSamplePoints(buttonRect)

  assert.equal(center.x, 50)
  assert.equal(center.y, 25)
})

test("clickable sample points: corner points are inset, not flush with the edges", () => {
  const buttonRect = rect({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100 })

  const points = computeClickableSamplePoints(buttonRect)
  const corners = points.slice(1)

  for (const corner of corners) {
    assert.notEqual(corner.x, buttonRect.left)
    assert.notEqual(corner.x, buttonRect.right)
    assert.notEqual(corner.y, buttonRect.top)
    assert.notEqual(corner.y, buttonRect.bottom)
  }
})

test("clickable sample points: majority threshold requires more than half of the 5 samples", () => {
  assert.equal(CLICKABLE_SAMPLE_MIN_HITS, 3)
  assert.ok(CLICKABLE_SAMPLE_MIN_HITS > 5 / 2)
})

// -----------------------------------
// computeOverlaySamplePoints() — the geometry behind detectBlockingOverlay()'s
// topmost-verification stage (added after FirstUser reported a "blocking
// overlay" that a real browser screenshot showed did not exist — a large
// decorative `position: absolute` element had passed the size/visibility
// checks without ever being confirmed as actually painted on top).
// -----------------------------------

test("overlay sample points: returns 3 points (center, upper-middle, lower-middle), all inside the rect", () => {
  const overlayRect = rect({ left: 100, top: 200, right: 1100, bottom: 900, width: 1000, height: 700 })

  const points = computeOverlaySamplePoints(overlayRect)

  assert.equal(points.length, 3)
  for (const point of points) {
    assert.ok(point.x >= overlayRect.left && point.x <= overlayRect.right, `x=${point.x} outside rect`)
    assert.ok(point.y >= overlayRect.top && point.y <= overlayRect.bottom, `y=${point.y} outside rect`)
  }
})

test("overlay sample points: center point is the exact geometric center", () => {
  const overlayRect = rect({ left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 })

  const [center] = computeOverlaySamplePoints(overlayRect)

  assert.equal(center.x, 640)
  assert.equal(center.y, 400)
})

test("overlay sample points: upper- and lower-middle points are inset, not flush with the top/bottom edges", () => {
  const overlayRect = rect({ left: 0, top: 0, right: 1280, bottom: 800, width: 1280, height: 800 })

  const [, upperMiddle, lowerMiddle] = computeOverlaySamplePoints(overlayRect)

  assert.notEqual(upperMiddle.y, overlayRect.top)
  assert.notEqual(lowerMiddle.y, overlayRect.bottom)
})

test("overlay topmost threshold: majority requires more than half of the 3 samples", () => {
  assert.equal(OVERLAY_TOPMOST_SAMPLE_MIN_HITS, 2)
  assert.ok(OVERLAY_TOPMOST_SAMPLE_MIN_HITS > 3 / 2)
})
