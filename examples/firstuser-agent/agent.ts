/**
 * FirstUser — AI-powered first-time user agent.
 *
 * Solari provides the browser.
 * OpenAI provides the reasoning.
 *
 * FirstUser behaves like a thoughtful first-time visitor:
 * observes → decides → acts → observes again.
 *
 * HARD RULE:
 * FirstUser never fills or submits forms.
 *
 * This module contains the core agent, extracted so it can be invoked
 * programmatically (by the CLI in index.ts, or by the API server) instead
 * of only running as a top-level script. The exploration logic, evidence
 * rules, final evaluator, and LLM budget are unchanged from the original
 * prototype — only the invocation shape changed.
 */

import { Solari } from "@solarisdk/browser"
import OpenAI from "openai"
import { mkdir, writeFile } from "node:fs/promises"

const solari = new Solari({
  apiKey: process.env.SOLARI_API_KEY!,
})

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
})

export const MAX_STEPS = 6

export const MAX_LLM_CALLS = 12

/**
 * Optional research safeguards, surfaced in the UI as plain checkboxes.
 * All default to `true` (the current, safest FirstUser behavior) when
 * omitted entirely, so existing callers (e.g. the CLI in index.ts) that
 * don't pass this field are unaffected.
 */
export interface ResearchRules {
  /** Ground evidence only in what was actually observed; no outside knowledge or invented facts. */
  evidenceOnly?: boolean
  /** Explore thoroughly enough to understand the core product before deciding no more info is needed. */
  exploreBeforeDeciding?: boolean
  /** Permit an "uncertain" final decision instead of forcing a yes/no when evidence is incomplete. */
  allowUncertain?: boolean
}

export interface FirstUserConfig {
  websiteUrl: string
  persona: string
  goal: string
  researchRules?: ResearchRules
}

export interface EvidenceItem {
  claim: string
  why_it_matters: string
  source: string
  observed_at_step: number
}

export interface ActionRecord {
  step: number
  action: string
  target_index: number | null
  dom_index: number | null
  target_text: string | null
  target_href: string | null
  reason: string
  url_after_action: string
}

export interface FinalDecision {
  decision: "yes" | "no" | "uncertain" | "incomplete"
  confidence?: "low" | "medium" | "high"
  evidence_found?: string[]
  evidence_missing?: string[]
  what_helped?: string[]
  what_blocked?: string[]
  reason: string
  recommendation?: string
}

export interface FirstUserResult {
  website: string
  persona: string
  goal: string
  journey: ActionRecord[]
  evidence: EvidenceItem[]
  decision: FinalDecision | null
  stoppedForLlmBudget: boolean
  llmUsage: { calls: number; maxCalls: number }
}

/**
 * Raw geometry/style for one fixed- or absolute-positioned element,
 * collected browser-side in observe(). Kept as plain data (no DOM
 * references) so the blocking-overlay decision below can run — and be
 * unit-tested — without a browser.
 */
export interface Rect {
  left: number
  top: number
  right: number
  bottom: number
  width: number
  height: number
}

export interface OverlayCandidate {
  position: string
  display: string
  visibility: string
  opacity: number
  rect: Rect
  text: string
  /**
   * How many of computeOverlaySamplePoints()'s interior sample points
   * report this element (or a descendant of it) as the topmost element
   * via elementFromPoint(). Confirms the element is actually painted on
   * top of what the visitor sees — a large `position: fixed`/`absolute`
   * element in the DOM is not, by itself, evidence of that.
   */
  topmostHits: number
}

/** A button/link belonging to the detected overlay — a safe, unambiguous click target. */
export interface OverlayAction {
  type: "button" | "link"
  text: string
  /** Matches the `index` FirstUser already uses as `target_index` for this element type. */
  index: number
}

export interface BlockingOverlayInfo {
  present: boolean
  /** Fraction (0–1) of the viewport area the overlay visually covers. */
  coverageRatio?: number
  /** Short, verbatim text observed inside the overlay. Never invented. */
  text?: string
  /** Internal only — used to find overlay-local controls; stripped before the observation reaches the LLM. */
  rect?: Rect
  /** Buttons/links that belong to the overlay itself, safe to target directly. */
  actions?: OverlayAction[]
}

/**
 * A first-time visitor notices an overlay when it materially obstructs
 * the page — not merely because *something* is fixed/absolute on screen
 * (sticky nav, small notices, chat bubbles are common and expected).
 * "Materially" is approximated here as covering a meaningful share of
 * the viewport; the exact threshold is a judgment call, not a precise
 * boundary.
 */
const MATERIAL_OVERLAY_COVERAGE_RATIO = 0.15

/**
 * How many of computeOverlaySamplePoints()'s 3 interior points must
 * report the candidate (or a descendant of it) as topmost before it can
 * be treated as a genuine blocking overlay. A majority (2 of 3) tolerates
 * one off-target sample while still requiring real evidence the element
 * is actually painted on top across its visible area — mirroring
 * CLICKABLE_SAMPLE_MIN_HITS's majority-of-samples approach below.
 */
export const OVERLAY_TOPMOST_SAMPLE_MIN_HITS = 2

/**
 * Interior points to hit-test when confirming a geometry-qualifying
 * candidate is actually topmost: center, upper-middle, lower-middle.
 * Deliberately avoids the candidate's edges (anti-aliasing/sub-pixel
 * rounding at an exact boundary is an unreliable sample) and avoids
 * relying on a single center point, which alone can't tell "large and
 * covers this area" apart from "large but only a strip of it is really
 * on top."
 */
export function computeOverlaySamplePoints(rect: Rect): Array<{ x: number; y: number }> {
  return [
    { x: rect.left + rect.width / 2, y: rect.top + rect.height * 0.5 },
    { x: rect.left + rect.width / 2, y: rect.top + rect.height * 0.25 },
    { x: rect.left + rect.width / 2, y: rect.top + rect.height * 0.75 },
  ]
}

/**
 * Pure geometry decision: given the fixed/absolute-positioned candidates
 * observed on the page, is one of them a material blocking overlay?
 * Deliberately has no opinion on WHAT the overlay is (cookie modal vs.
 * login wall vs. promo) — that classification is left to the LLM, which
 * can read the overlay's actual text and the page's buttons/links.
 *
 * Geometry (size/visibility) is only the first stage. A `position:
 * fixed`/`absolute` element can be large and fully opaque while still
 * being purely decorative — e.g. a hero section's illustration wrapper
 * — if it sits behind the page's real content in paint order. Requiring
 * `topmostHits` (computed in the browser via elementFromPoint(), see
 * observe()) keeps that class of element from being reported as a
 * blocking overlay it never actually was.
 */
export function detectBlockingOverlay(
  candidates: OverlayCandidate[],
  viewport: { width: number; height: number }
): BlockingOverlayInfo {
  const viewportArea = viewport.width * viewport.height
  if (viewportArea <= 0) return { present: false }

  let best: { coverageRatio: number; text: string; rect: Rect } | null = null

  for (const candidate of candidates) {
    if (candidate.display === "none" || candidate.visibility === "hidden") continue
    if (candidate.opacity === 0) continue
    if (candidate.rect.width <= 0 || candidate.rect.height <= 0) continue

    const visibleWidth = Math.max(
      0,
      Math.min(candidate.rect.right, viewport.width) - Math.max(candidate.rect.left, 0)
    )
    const visibleHeight = Math.max(
      0,
      Math.min(candidate.rect.bottom, viewport.height) - Math.max(candidate.rect.top, 0)
    )
    const coverageRatio = (visibleWidth * visibleHeight) / viewportArea

    if (coverageRatio < MATERIAL_OVERLAY_COVERAGE_RATIO) continue
    if (candidate.topmostHits < OVERLAY_TOPMOST_SAMPLE_MIN_HITS) continue

    if (!best || coverageRatio > best.coverageRatio) {
      best = { coverageRatio, text: candidate.text, rect: candidate.rect }
    }
  }

  if (!best) return { present: false }

  return {
    present: true,
    coverageRatio: Math.round(best.coverageRatio * 100) / 100,
    text: best.text,
    rect: best.rect,
  }
}

/**
 * Which of the page's buttons/links actually belong to the detected
 * overlay — computed by rect overlap, not DOM containment, so it still
 * works if a dismiss control is a sibling of the backdrop rather than a
 * descendant (a common pattern in portal-based modal libraries).
 *
 * This is a hint, not a guarantee: a background button that happens to
 * sit in the same screen region could be misclassified as an overlay
 * action. isTargetClickable()'s runtime elementFromPoint() check is the
 * actual safety net that prevents a misclassified target from ever being
 * clicked incorrectly — this function only narrows the LLM's choices.
 */
export function findOverlayActions(
  overlay: BlockingOverlayInfo,
  buttons: Array<{ index: number; text: string; rect: Rect }>,
  links: Array<{ index: number; text: string; rect: Rect }>
): OverlayAction[] {
  if (!overlay.present || !overlay.rect) return []

  const overlayRect = overlay.rect

  const overlaps = (rect: Rect) =>
    rect.left < overlayRect.right &&
    rect.right > overlayRect.left &&
    rect.top < overlayRect.bottom &&
    rect.bottom > overlayRect.top

  const buttonActions: OverlayAction[] = buttons
    .filter((button) => overlaps(button.rect))
    .map((button) => ({ type: "button", text: button.text, index: button.index }))

  const linkActions: OverlayAction[] = links
    .filter((link) => overlaps(link.rect))
    .map((link) => ({ type: "link", text: link.text, index: link.index }))

  return [...buttonActions, ...linkActions]
}

/**
 * Which points across an element's bounding box to hit-test before
 * concluding it is (or isn't) actually clickable. Center-only sampling
 * proved too fragile in practice — a real run against Notion showed
 * elementFromPoint() at the exact center reporting a fully on-screen,
 * visually unobstructed button as covered (confirmed via screenshot).
 * Sampling the center plus four inset corners and requiring a majority
 * hit tolerates a single off-target sample (sub-pixel rounding, a
 * decorative icon at the exact center) while still failing a genuinely
 * covered element, which would miss most or all samples.
 */
export const CLICKABLE_SAMPLE_MIN_HITS = 3

export function computeClickableSamplePoints(rect: Rect): Array<{ x: number; y: number }> {
  const insetX = rect.width * 0.2
  const insetY = rect.height * 0.2

  return [
    { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 },
    { x: rect.left + insetX, y: rect.top + insetY },
    { x: rect.right - insetX, y: rect.top + insetY },
    { x: rect.left + insetX, y: rect.bottom - insetY },
    { x: rect.right - insetX, y: rect.bottom - insetY },
  ]
}

export class LlmBudgetExceededError extends Error {
  constructor(maxCalls: number) {
    super(
      `FirstUser LLM budget exceeded: maximum ${maxCalls} OpenAI calls per run.`
    )
    this.name = "LlmBudgetExceededError"
  }
}

/** Raised when the target site cannot be reached at all (DNS, TLS, refused, timeout). */
export class WebsiteUnreachableError extends Error {
  constructor(websiteUrl: string, cause: unknown) {
    super(`FirstUser could not reach ${websiteUrl}.`)
    this.name = "WebsiteUnreachableError"
    this.cause = cause as Error | undefined
  }
}

export interface FirstUserHooks {
  onLog?: (message: string) => void
  onSessionStarted?: (config: FirstUserConfig) => void
  onNavigation?: (url: string) => void
  onObservation?: (info: {
    step: number
    url: string
    title: string
    screenshotPath: string
  }) => void
  onEvidence?: (info: {
    step: number
    found: Array<{ claim: string; why_it_matters: string; source: string }>
    missing: Array<{ claim: string; why_it_matters: string }>
  }) => void
  onAction?: (record: ActionRecord) => void
  onStopped?: (reason: string) => void
  /** Fired once exploration ends and the final evaluator is about to run. */
  onEvaluating?: () => void
  onBudgetWarning?: (message: string) => void
  onCompleted?: (result: FirstUserResult) => void
  onError?: (error: Error) => void
}

export interface RunOptions {
  /** Directory screenshots are written to. Defaults to "screenshots". */
  screenshotDir?: string
}

/**
 * Closes the shared Solari client. `browser.close()` (called after every
 * run) releases the individual session; this releases the client's own
 * resources and should only be called once, when the whole process exits.
 */
export async function closeSolariClient() {
  await solari.close()
}

export async function runFirstUserTest(
  config: FirstUserConfig,
  hooks: FirstUserHooks = {},
  options: RunOptions = {}
): Promise<FirstUserResult> {
  const testConfig = {
    websiteUrl: config.websiteUrl,
    persona: config.persona,
    goal: config.goal,
  }

  // Research rules default to ON (the current, safest behavior) whenever
  // omitted or not explicitly `false` — see ResearchRules for what each
  // one guards.
  const researchRules = {
    evidenceOnly: config.researchRules?.evidenceOnly !== false,
    exploreBeforeDeciding: config.researchRules?.exploreBeforeDeciding !== false,
    allowUncertain: config.researchRules?.allowUncertain !== false,
  }

  // Prompt notes only ever ADD an override — when a rule is left ON
  // (the default), the exploration/evaluation prompts are byte-identical
  // to the pre-existing FirstUser prompts.
  const evidenceOnlyOverrideNote = researchRules.evidenceOnly
    ? ""
    : `

NOTE: For this session, "Evidence only" has been turned OFF by the user.
You may supplement the current observation with reasonable general or
domain knowledge when it alone is insufficient. If you do, phrase it
clearly as an inference or general expectation (e.g. "likely," "typically")
rather than stating it as something the page itself said.`

  const exploreBeforeDecidingOverrideNote = researchRules.exploreBeforeDeciding
    ? ""
    : `

NOTE: For this session, "Explore before deciding" has been turned OFF by
the user. You do not need to fully explore the page or additional
sections before deciding you have enough information — if the current
evidence reasonably addresses the goal, you may choose "none" sooner
rather than continuing to explore.`

  const allowUncertainOverrideNote = researchRules.allowUncertain
    ? ""
    : `

NOTE: For this session, "Allow UNCERTAIN" has been turned OFF by the
user. Prefer a definitive "yes" or "no" decision using your best
judgment from the available evidence, even if some information is
missing. Only return "uncertain" if there is truly no relevant evidence
at all to reason from.`

  const screenshotDir = options.screenshotDir ?? "screenshots"

  const log = (message: string) => hooks.onLog?.(message)

  let llmCallCount = 0

  // One call is always reserved for evaluateJourney(), so exploration
  // (the combined evidence + action reasoning call) may use at most this many.
  const MAX_EXPLORATION_LLM_CALLS = MAX_LLM_CALLS - 1

  function explorationBudgetReached() {
    return llmCallCount >= MAX_EXPLORATION_LLM_CALLS
  }

  function logExplorationBudgetReached() {
    const reserved = MAX_LLM_CALLS - MAX_EXPLORATION_LLM_CALLS
    log("--- EXPLORATION BUDGET REACHED ---")
    log(`Exploration stopped after ${MAX_EXPLORATION_LLM_CALLS} LLM calls.`)
    log(`${reserved} LLM call reserved for final evaluation.`)
    hooks.onBudgetWarning?.(
      `Exploration stopped after ${MAX_EXPLORATION_LLM_CALLS} LLM calls; ${reserved} reserved for final evaluation.`
    )
  }

  async function callOpenAI(label: string, params: any) {
    if (llmCallCount >= MAX_LLM_CALLS) {
      throw new LlmBudgetExceededError(MAX_LLM_CALLS)
    }

    llmCallCount += 1

    log(`LLM call ${llmCallCount}/${MAX_LLM_CALLS} — ${label}`)

    return openai.responses.create(params)
  }

  const explorationMemory = {
    visitedUrls: [] as string[],
    visitedTargets: [] as any[],
    exploredScrollPositions: [] as number[],
  }

  function rememberUrl(url: string) {
    if (!explorationMemory.visitedUrls.includes(url)) {
      explorationMemory.visitedUrls.push(url)
    }
  }

  const evidence: EvidenceItem[] = []

  await mkdir(screenshotDir, { recursive: true })

  async function saveScreenshot(page: any, filename: string) {
    const png = await page.screenshot()
    const path = `${screenshotDir}/${filename}`
    await writeFile(path, png)
    log(`Saved screenshot: ${path}`)
    return path
  }

  // -----------------------------------
  // FINAL DECISION LAYER
  // -----------------------------------

  async function evaluateJourney(observations: any[], actionHistory: ActionRecord[]) {
    log("--- FINAL EVALUATION ---")

    const evaluation = await callOpenAI("final evaluation", {
      model: "gpt-5",

      input: `
You are the final reasoning layer of FirstUser.

An AI visitor has just explored a website.

Your job is to determine whether the visitor can answer their goal
using ONLY evidence that was actually available during this session.

WEBSITE UNDER TEST:
${testConfig.websiteUrl}

VISITOR PERSONA:
${testConfig.persona}

VISITOR GOAL:
${testConfig.goal}

IMPORTANT RULES:

1. Use ONLY evidence contained in the observations below.

2. Do NOT use outside knowledge.

3. Do NOT invent missing numbers.

4. Do NOT assume a recovery rate, expected recovered amount,
   conversion rate, price, feature, guarantee, or other fact
   unless it was actually visible on the website.

5. Distinguish carefully between:
   - failed payment volume
   - recovered payment value
   - Reclaim's fee
   - expected financial return

6. If the website does not provide enough evidence to answer
   the visitor's goal confidently, the correct decision is:
   "uncertain".

7. Do not force a yes/no answer.

8. Missing evidence is itself an important finding.

9. Base the evaluation on what the visitor actually encountered,
   not what you know about the website from outside this session.

10. Treat the explicit evidence collection as a structured record
    of what the visitor identified during the journey.

11. Evidence must still be grounded in the original observations.
    Do not trust an evidence claim if it contradicts the observation.

12. A dismissible overlay the visitor encountered and handled (e.g. a
    cookie consent modal that was accepted or rejected) is normal
    first-time-visitor experience, not a negative signal — do not let it
    lower confidence or the decision on its own. Only treat an overlay
    as a blocker/evidence gap if it genuinely prevented the visitor from
    reaching content needed to answer the goal (e.g. a login wall or
    paywall blocking core information).
${allowUncertainOverrideNote}

SESSION OBSERVATIONS:

${JSON.stringify(observations, null, 2)}

EXPLICIT EVIDENCE COLLECTED DURING THE JOURNEY:

${JSON.stringify(evidence, null, 2)}

ACTION HISTORY:

${JSON.stringify(actionHistory, null, 2)}

Return ONLY valid JSON:

{
  "decision": "yes" | "no" | "uncertain",
  "confidence": "low" | "medium" | "high",
  "evidence_found": [
    "specific evidence found on the website"
  ],
  "evidence_missing": [
    "specific information that was needed but not found"
  ],
  "what_helped": [
    "things that helped the visitor evaluate the goal"
  ],
  "what_blocked": [
    "things that prevented the visitor from confidently answering"
  ],
  "reason": "concise explanation of the decision",
  "recommendation": "specific opportunity based on the evidence"
}
`,
    })

    let result

    try {
      result = JSON.parse(evaluation.output_text)
    } catch (error) {
      log("--- FINAL EVALUATION ERROR ---")
      log("AI returned invalid JSON.")
      log(evaluation.output_text)
      return null
    }

    log("FIRSTUSER FINAL DECISION")
    log(JSON.stringify(result, null, 2))

    return result
  }

  const browser = await solari.launch()

  try {
    let page: any

    try {
      page = await browser.newPage()
      await page.goto(testConfig.websiteUrl)
    } catch (error) {
      throw new WebsiteUnreachableError(testConfig.websiteUrl, error)
    }

    rememberUrl(page.url())
    hooks.onSessionStarted?.(testConfig)
    hooks.onNavigation?.(page.url())

    // -----------------------------------
    // OBSERVE THE CURRENT PAGE
    // -----------------------------------

    async function observe() {
      const viewport = await page.evaluate(() => ({
        width: window.innerWidth,
        height: window.innerHeight,
        scrollY: window.scrollY,
        documentHeight: document.documentElement.scrollHeight,
      }))

      const buttons = await page
        .locator("button")
        .evaluateAll((elements: any[]) =>
          elements
            .map((element, domIndex) => {
              const rect = element.getBoundingClientRect()
              const style = window.getComputedStyle(element)

              const visible =
                style.display !== "none" &&
                style.visibility !== "hidden" &&
                rect.width > 0 &&
                rect.height > 0 &&
                rect.bottom > 0 &&
                rect.top < window.innerHeight &&
                rect.right > 0 &&
                rect.left < window.innerWidth

              return {
                dom_index: domIndex,
                text: element.innerText.trim(),
                visible,
                rect: { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom, width: rect.width, height: rect.height },
              }
            })
            .filter((item) => item.visible)
            .map((item, visibleIndex) => ({
              index: visibleIndex,
              dom_index: item.dom_index,
              text: item.text,
              visible: item.visible,
              rect: item.rect,
            }))
        )

      const links = await page
        .locator("a")
        .evaluateAll((elements: any[]) =>
          elements
            .map((element, domIndex) => {
              const rect = element.getBoundingClientRect()
              const style = window.getComputedStyle(element)

              const visible =
                style.display !== "none" &&
                style.visibility !== "hidden" &&
                rect.width > 0 &&
                rect.height > 0 &&
                rect.bottom > 0 &&
                rect.top < window.innerHeight &&
                rect.right > 0 &&
                rect.left < window.innerWidth

              return {
                dom_index: domIndex,
                text: element.innerText.trim(),
                href: element.href,
                visible,
                rect: { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom, width: rect.width, height: rect.height },
              }
            })
            .filter((item) => item.visible)
            .map((item, visibleIndex) => ({
              index: visibleIndex,
              dom_index: item.dom_index,
              text: item.text,
              href: item.href,
              visible: item.visible,
              rect: item.rect,
            }))
        )

      const expandables = await page
        .locator("summary, button[aria-expanded], button[aria-controls]")
        .evaluateAll((elements: any[]) =>
          elements
            .map((element, domIndex) => {
              const rect = element.getBoundingClientRect()
              const style = window.getComputedStyle(element)
              const details = element.closest("details")

              const visible =
                style.display !== "none" &&
                style.visibility !== "hidden" &&
                rect.width > 0 &&
                rect.height > 0 &&
                rect.bottom > 0 &&
                rect.top < window.innerHeight &&
                rect.right > 0 &&
                rect.left < window.innerWidth

              return {
                dom_index: domIndex,
                text: element.innerText.trim(),
                visible,
                expanded: details
                  ? details.open
                  : element.getAttribute("aria-expanded") === "true",
              }
            })
            .filter((item) => item.visible && item.text)
            .map((item, visibleIndex) => ({
              index: visibleIndex,
              dom_index: item.dom_index,
              text: item.text,
              visible: item.visible,
              expanded: item.expanded,
            }))
        )

      const visibleText = await page.locator("body").evaluate((body) => {
        const elements = Array.from(
          body.querySelectorAll("h1, h2, h3, p, li, td, th, summary, details")
        )

        const texts = elements
          .filter((element) => {
            const rect = element.getBoundingClientRect()
            const style = window.getComputedStyle(element)

            return (
              style.display !== "none" &&
              style.visibility !== "hidden" &&
              rect.width > 0 &&
              rect.height > 0 &&
              rect.bottom > 0 &&
              rect.top < window.innerHeight &&
              rect.right > 0 &&
              rect.left < window.innerWidth
            )
          })
          .map((element) => element.innerText.replace(/\s+/g, " ").trim())
          .filter(Boolean)

        return [...new Set(texts)].join("\n")
      })

      const formFieldCount = await page.locator("input, textarea, select").count()

      // Raw geometry/style only — no threshold or "is this an overlay"
      // judgment here. That decision is made by the pure, unit-testable
      // detectBlockingOverlay() below, entirely outside the browser.
      //
      // topmostHits is the one exception: it requires a live DOM
      // (elementFromPoint()), so it has to be computed here rather than
      // in detectBlockingOverlay(). The 3 sample points mirror
      // computeOverlaySamplePoints() exactly — duplicated inline because
      // a page.evaluate() callback runs in the browser and can't close
      // over an outer Node.js function. Only computed for fixed/absolute
      // elements (the only ones that can ever become overlay candidates)
      // so this doesn't run elementFromPoint() for every element on the page.
      const overlayCandidates: OverlayCandidate[] = await page.evaluate(() =>
        Array.from(document.querySelectorAll("body *"))
          .map((element: any) => {
            const style = window.getComputedStyle(element)
            const rect = element.getBoundingClientRect()
            const position = style.position

            let topmostHits = 0

            if ((position === "fixed" || position === "absolute") && rect.width > 0 && rect.height > 0) {
              const samplePoints = [
                { x: rect.left + rect.width / 2, y: rect.top + rect.height * 0.5 },
                { x: rect.left + rect.width / 2, y: rect.top + rect.height * 0.25 },
                { x: rect.left + rect.width / 2, y: rect.top + rect.height * 0.75 },
              ]

              for (const point of samplePoints) {
                const inBounds =
                  point.x >= 0 && point.y >= 0 && point.x <= window.innerWidth && point.y <= window.innerHeight
                if (!inBounds) continue

                const topElement = document.elementFromPoint(point.x, point.y)
                if (topElement && (topElement === element || element.contains(topElement))) {
                  topmostHits++
                }
              }
            }

            return {
              position,
              display: style.display,
              visibility: style.visibility,
              opacity: parseFloat(style.opacity || "1"),
              rect: {
                left: rect.left,
                top: rect.top,
                right: rect.right,
                bottom: rect.bottom,
                width: rect.width,
                height: rect.height,
              },
              text: (element.innerText || "").replace(/\s+/g, " ").trim().slice(0, 300),
              topmostHits,
            }
          })
          .filter((item) => item.position === "fixed" || item.position === "absolute")
      )

      const blockingOverlay = detectBlockingOverlay(overlayCandidates, {
        width: viewport.width,
        height: viewport.height,
      })

      const overlayActions = findOverlayActions(blockingOverlay, buttons, links)

      // The LLM-facing observation only needs to know an overlay is
      // present, roughly how much it covers, what it says, and which
      // controls belong to it — not raw pixel rects (internal-only, used
      // above to compute overlayActions).
      const blockingOverlayForObservation = blockingOverlay.present
        ? {
            present: true as const,
            coverageRatio: blockingOverlay.coverageRatio,
            text: blockingOverlay.text,
            actions: overlayActions,
          }
        : { present: false as const }

      const buttonsForObservation = buttons.map(({ rect, ...rest }) => rest)
      const linksForObservation = links.map(({ rect, ...rest }) => rest)

      return {
        url: page.url(),
        title: await page.title(),
        viewport,
        blockingOverlay: blockingOverlayForObservation,

        headings: await page
          .locator("h1, h2")
          .evaluateAll((elements: any[]) =>
            elements
              .map((element) => {
                const rect = element.getBoundingClientRect()
                const style = window.getComputedStyle(element)

                const visible =
                  style.display !== "none" &&
                  style.visibility !== "hidden" &&
                  rect.width > 0 &&
                  rect.height > 0 &&
                  rect.bottom > 0 &&
                  rect.top < window.innerHeight &&
                  rect.right > 0 &&
                  rect.left < window.innerWidth

                return {
                  text: element.innerText.trim(),
                  visible,
                }
              })
              .filter((item) => item.visible)
              .map((item) => item.text)
          ),

        buttons: buttonsForObservation,
        links: linksForObservation,
        expandables,
        visibleText: visibleText.slice(0, 4000),
        pageTextLength: visibleText.length,
        hasForm: formFieldCount > 0,
      }
    }

    // -----------------------------------
    // INTERACTION SAFETY
    //
    // observe()'s visibility checks only confirm a target is on-screen and
    // not display:none/hidden — they don't know whether something else
    // (e.g. a blocking overlay) now sits on top of it. Before clicking,
    // confirm the target is actually the topmost element at its own
    // center point, so an overlay can never silently absorb a click while
    // FirstUser believes it interacted with the page.
    // -----------------------------------

    async function isTargetClickable(selector: string, domIndex: number) {
      // Sample multiple points across the target's bounding box, not just
      // its exact center. A real run against Notion showed a single
      // center-point elementFromPoint() check reporting a fully on-screen,
      // visually unobstructed "Reject all" button as occluded (confirmed
      // via screenshot) — a single sample is fragile to sub-pixel
      // rounding, a decorative icon/pseudo-element sitting exactly at the
      // geometric center, or compositing quirks on a freshly-appeared
      // overlay. See computeClickableSamplePoints()/CLICKABLE_SAMPLE_MIN_HITS
      // for the (unit-tested) geometry and threshold this relies on.
      const rect = await page.evaluate(
        ({ selector, domIndex }: { selector: string; domIndex: number }) => {
          const el = document.querySelectorAll(selector)[domIndex]
          if (!el) return null

          const r = el.getBoundingClientRect()
          if (r.width <= 0 || r.height <= 0) return null

          return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height }
        },
        { selector, domIndex }
      )

      if (!rect) return false

      const points = computeClickableSamplePoints(rect)

      const hits = await page.evaluate(
        ({
          selector,
          domIndex,
          points,
        }: {
          selector: string
          domIndex: number
          points: Array<{ x: number; y: number }>
        }) => {
          const el = document.querySelectorAll(selector)[domIndex]
          if (!el) return 0

          let hitCount = 0

          for (const point of points) {
            const inBounds =
              point.x >= 0 && point.y >= 0 && point.x <= window.innerWidth && point.y <= window.innerHeight
            if (!inBounds) continue

            const topElement = document.elementFromPoint(point.x, point.y)
            if (topElement && (topElement === el || el.contains(topElement) || topElement.contains(el))) {
              hitCount++
            }
          }

          return hitCount
        },
        { selector, domIndex, points }
      )

      return hits >= CLICKABLE_SAMPLE_MIN_HITS
    }

    // -----------------------------------
    // KEEP TRACK OF USER JOURNEY
    // -----------------------------------

    const actionHistory: ActionRecord[] = []
    const observations: any[] = []

    // -----------------------------------
    // COMBINED EVIDENCE EXTRACTION + ACTION REASONING
    //
    // A single LLM call performs both responsibilities that previously
    // required two separate round-trips:
    //   1. extract goal-relevant evidence from the CURRENT observation
    //   2. decide the single most useful next browser action
    // Both responsibilities read the same observation; evidence extraction
    // does not depend on which action is chosen, and the action decision
    // does not depend on the evidence extraction succeeding.
    // -----------------------------------

    async function exploreAndReason(observation: any, step: number) {
      log(`--- STEP ${step}: EXPLORATION REASONING + EVIDENCE ---`)

      const response = await callOpenAI("exploration reasoning + evidence", {
        model: "gpt-5",

        input: `
You are FirstUser, an AI agent behaving like a thoughtful first-time user exploring a website.

WEBSITE UNDER TEST:
${testConfig.websiteUrl}

VISITOR PERSONA:
${testConfig.persona}

VISITOR GOAL:
${testConfig.goal}

You have two responsibilities in this single response. Perform both,
independently, using the CURRENT PAGE OBSERVATION provided near the end
of this prompt.

===========================================
VISITOR-STATE AWARENESS (applies to both responsibilities)
===========================================

Before reasoning about page content, check the observation's
"blockingOverlay" field. THIS CHECK COMES FIRST, before either
responsibility below, and if it applies it takes priority over normal
exploration this step — see the "OVERLAY PRIORITY GATE" at the start of
Responsibility 2's action reasoning for exactly how that priority is
enforced.

First decide whether something is materially obstructing this
first-time visitor's ability to see or interact with the actual page —
for example a cookie consent modal, a newsletter/promotional popup, an
app-download interstitial, a region/age gate, a login wall, a paywall,
or a CAPTCHA/security challenge. Ordinary chat bubbles, sticky
navigation, small non-blocking notices, and tooltips are NOT blockers —
"blockingOverlay.present" only becomes true when something covers a
meaningful share of the viewport, but still use judgment: a small
element is not a blocker even if flagged.

If something IS blocking:

1. Record it as evidence (Responsibility 1) — e.g. "A cookie consent
   modal appeared on initial page load and obscured part of the page."
   Describe only what "blockingOverlay.text" and the page actually show;
   never invent details about the overlay.

2. Identify safe controls. "blockingOverlay.actions" lists the specific
   buttons/links that belong to the overlay itself, each with the exact
   "index" you would use as "target_index" to click it. Read their
   "text" to judge which is safe (see policy below). If
   "blockingOverlay.actions" is empty, there is no known safe control —
   treat this the same as "no safe resolution available" below.

3. Choose the safe resolution action (Responsibility 2 — this is what
   "target_index"/"action" in your response must reflect this step):
   - Cookie/consent UI: prefer "Reject all"/"Decline"/"Essential only"
     over "Accept all" when a clear reject-style option exists among
     "blockingOverlay.actions"; only accept if that is the only option
     and the modal is genuinely blocking; do not open a detailed
     preferences screen unless it is directly relevant to the research
     goal.
   - Newsletter/promotional popups and interstitials: prefer closing or
     dismissing them (e.g. "Close", "×", "No thanks").
   - Login/signup walls: do NOT create an account, log in, or invent
     credentials — these are never a safe "blockingOverlay.actions"
     choice. Record that the experience is gated and pick a different
     action (or "none") if no safe path forward exists.
   - Paywalls: do NOT attempt to bypass them. Record the limitation.
   - CAPTCHAs/security challenges: do NOT attempt to circumvent them.
     Record the limitation.
   - Age/eligibility/region gates: do not fabricate eligibility. Treat
     them as a visitor-state constraint and record what happened.

Never bypass authentication, paywalls, CAPTCHAs, security controls, or
other access restrictions, no matter how blocking they are. A blocking
overlay you can safely dismiss (like a cookie banner) is normal
first-time-visitor friction, not a website failure. Only a genuine
access restriction that prevents reaching content needed for the
visitor's goal (e.g. a login wall blocking core information) should be
treated as an evidence gap.

===========================================
RESPONSIBILITY 1 — EVIDENCE EXTRACTION
===========================================

You are also the evidence extraction layer of FirstUser.

Your job here is NOT to make the final decision.

Your job is to identify only the pieces of information in the
CURRENT VIEWPORT that materially help answer the visitor's goal.

IMPORTANT RULES:

1. Use ONLY information contained in the current observation.

2. Do NOT use outside knowledge.

3. Do NOT infer facts that are not explicitly supported.

4. Do NOT turn generic marketing language into quantified evidence.

5. Extract concrete, decision-relevant claims.

6. If a piece of information is potentially relevant but does not
   actually provide useful evidence, do not include it.

7. Missing information should NOT be invented.

8. Only identify information as "missing" if its absence directly
prevents the visitor from accomplishing the stated goal.

9. Do NOT generate a general due-diligence checklist.

10. Do NOT list information merely because it would be nice to know.

11. Missing evidence must have a direct causal relationship to the
visitor being unable to make progress toward their goal.

12. Prefer identifying the minimum critical missing information.

13. Evidence is cumulative across the visitor journey.

14. Do NOT describe previously discovered evidence as missing merely
because it is not visible in the current viewport.

15. The current viewport is used to discover NEW evidence.
    Previously collected evidence remains available to the visitor.
${evidenceOnlyOverrideNote}

Ask yourself:

"Could the visitor reasonably accomplish their stated goal without
this information?"

If yes, do NOT classify it as critical missing evidence.

The evidence you extract for this responsibility must be grounded ONLY
in the CURRENT PAGE OBSERVATION shown near the end of this prompt. It
must NOT depend on, or be influenced by, which action you choose for
Responsibility 2 below.

CURRENT STEP:
${step}

EVIDENCE ALREADY COLLECTED FROM EARLIER STEPS:

${JSON.stringify(evidence, null, 2)}

===========================================
RESPONSIBILITY 2 — NEXT ACTION REASONING
===========================================

Your job here is to explore the website naturally and determine whether you can accomplish the visitor goal above.

The visitor goal should guide your decisions.

Only investigate information or take actions when they help you make progress toward the visitor goal.

You are currently on STEP ${step} of a browsing journey.

You may take up to ${MAX_STEPS} actions total.

IMPORTANT MENTAL MODEL:

A website's navigation menu is NOT a complete map of its information
architecture.

Meaningful evidence toward the visitor's goal often lives outside the
navigation entirely, for example:

- below the fold on the current page
- in landing-page sections
- in cards or feature grids
- in testimonials or other social proof
- near calls-to-action
- in footer content

Treat the current page itself as a source of evidence, not just a
launchpad into the navigation. Before assuming you need to navigate
elsewhere, consider whether the current page still has unexplored
content that could contain relevant evidence.

When deciding your next action, reason in roughly this order:

0. OVERLAY PRIORITY GATE — check this BEFORE steps 1-5. If
   "blockingOverlay.present" is true and it materially obstructs the
   page:
   - If "blockingOverlay.actions" contains a safe control (per the
     policy in VISITOR-STATE AWARENESS above), your action THIS STEP
     MUST be "click_button" or "click_link" targeting that control's
     "index" as "target_index". Resolving it overrides scrolling,
     opening navigation menus, clicking product links, expanding
     content, or any other step-1-through-5 action — even if one of
     those looks more directly useful for the research goal.
   - If "blockingOverlay.actions" is empty or none of the listed
     controls are safe (e.g. only "Log in"/"Sign up" is offered, or this
     is a paywall/CAPTCHA with no dismiss control), do NOT attempt any
     of them. Record the limitation as evidence and fall through to
     steps 1-5 to continue research only where the obstruction doesn't
     prevent it — or choose "none" if nothing safe remains.
   - Only proceed to steps 1-5 below once the overlay has been addressed
     by one of the two outcomes above (resolved, or determined
     unsafe-to-resolve and recorded).
1. What useful information is already visible in the current viewport?
2. Is there likely to be meaningful information further down this page?
3. If so, is scrolling the best next action?
4. If not, is there a useful visible interaction (link, button, expandable)?
5. If the current page is no longer likely to produce useful evidence,
   navigate to another page.

This is a guide for reasoning, not a fixed sequence. If a navigation
destination (e.g. "Pricing") is clearly the fastest way to answer the
visitor's goal, choose it immediately — do not scroll first merely for
the sake of scrolling. Step 0 (the overlay gate) is the one exception to
this flexibility: when it applies, it is not optional.

For scroll actions:

- Choose a scroll amount between 300 and 900 pixels.
- Use smaller scrolls when you are near the bottom of the page.
- Use smaller scrolls when the current section may contain useful information that should not be skipped.
- Use larger scrolls when the current section appears complete and there is substantial unseen content.
- Prefer gradual exploration over jumping large distances.
- After every scroll, you will observe the page again.

Your previous actions are:

${JSON.stringify(actionHistory, null, 2)}

EXPLORATION MEMORY:

${JSON.stringify(explorationMemory, null, 2)}

EXPLORATION RULES:

1. Treat the exploration memory as the visitor's memory of what
   has already been investigated.

2. Do NOT repeat an action merely because it is relevant.

3. Before choosing an action, ask:
   "Have I already taken this action, and did it reveal anything new?"

4. If an action has already been taken and produced no useful new
   evidence, prefer another unexplored action.

5. If the current viewport contains unexplored content that could
   answer the visitor's goal, prefer exploring that content.

6. Scrolling is exploration, not failure.

7. Use "up" only when there is a specific reason to revisit
   previously seen content.

8. Do not click the same link or button repeatedly unless the
   page state has materially changed.

9. If repeated scrolling within the same section has not revealed
   meaningful new information, STOP exploring that section.

10. If the last two actions were scrolls and they moved between
    substantially similar page states without revealing useful
    new evidence, do NOT scroll again. Choose another unexplored
    destination or choose "none".

11. Do not oscillate between the same scroll positions.

12. If another unexplored informational destination is available
    and the current section is not producing useful evidence,
    prefer that destination.

13. If no remaining action is likely to produce meaningful new
    evidence, choose "none".

14. The objective is NOT to explore the entire website.

    The objective is to gather enough evidence to answer the goal.

15. Prefer the action with the highest expected information value
    toward answering the visitor's goal.

16. Stop exploring when the remaining uncertainty cannot reasonably
    be reduced through available website actions.
${exploreBeforeDecidingOverrideNote}

EVIDENCE DISCOVERED SO FAR:

(This is the same cumulative evidence list already shown above in
Responsibility 1 — repeated here only as reasoning context for the
action decision.)

${JSON.stringify(evidence, null, 2)}

IMPORTANT — VIEWPORT AWARENESS:

You are acting like a real first-time website visitor.

Only the content in CURRENT VIEWPORT should be treated as information
you currently know from the page.

Do not assume you have read content that is below or above the current
viewport.

Visible buttons, links, and expandable elements are the elements you can currently interact with.

If the current viewport does not contain enough useful information and
there is more page content to explore, use "scroll".

When scrolling:

- use "down" to explore new content below the current viewport
- use "up" only when returning to something previously seen
- do not scroll randomly
- prefer scrolling when it is likely to reveal information relevant
  to your questions

Before choosing an action, ask yourself:

"Will this action give me meaningfully new information?"

If yes, take it.
If no, choose another action or choose "none".

Do not click links or buttons that are not currently visible.

Use "toggle_expandable" when a visible question, FAQ item, or expandable
heading is likely to contain evidence relevant to the visitor's goal.
If the current viewport contains a relevant collapsed expandable element,
prefer toggling it before leaving that section.

Continue prioritizing:

- what the product does
- whether it is useful
- how it works
- cost
- trust
- what the next step is

Never fill or submit forms.

Never log in, sign up, check out, make a payment, create an account,
or enter personal/payment information.

IMPORTANT BEHAVIOR RULES:

1. Choose the SINGLE most useful next action for a first-time visitor.

2. HARD RULE: You NEVER fill in forms.

3. HARD RULE: You NEVER submit forms.

4. If a form or input field appears, do not interact with it.

5. Do not attempt login, signup, checkout, payment, account creation,
   or data entry.

6. Navigation links to informational sections such as:
   "How it works"
   "Pricing"
   "FAQ"
   "The problem"
   are generally safe to explore.

7. Do not choose a link whose href is the same as the current page URL
   unless clicking it is clearly expected to cause a meaningful state
   change such as opening a modal, accordion, menu, or other interactive
   state.

===========================================
CURRENT PAGE OBSERVATION (shared by both responsibilities above):
===========================================

${JSON.stringify(observation, null, 2)}

Buttons, links, and expandable elements are numbered by index.

Some elements may have the same visible text.

For example:

- Link index 2: "How it works"
- Link index 9: "How it works"

These may be different elements.

Choose the exact index you want.

Choose ONE action for Responsibility 2.

The action must be one of:

- click_button
- click_link
- toggle_expandable
- scroll
- none

Only choose an index that actually exists for that action type.

Return ONLY valid JSON combining both responsibilities:

{
  "evidence_found": [
    {
      "claim": "specific factual claim visible in the current viewport",
      "why_it_matters": "why this helps answer the visitor's goal",
      "source": "page section or visible context"
    }
  ],
  "evidence_missing": [
    {
      "claim": "specific information that appears necessary but is not available in the current viewport",
      "why_it_matters": "why this information is needed"
    }
  ],
  "action": "click_button" | "click_link" | "toggle_expandable" | "scroll" | "none",
  "target_index": number | null,
  "scroll_direction": "down" | "up" | null,
  "scroll_amount": number | null,
  "reason": "why this is the most useful next action"
}

If there is no relevant evidence, return empty arrays for evidence_found and evidence_missing.
`,
      })

      let parsed

      try {
        parsed = JSON.parse(response.output_text)
      } catch (error) {
        log("--- EXPLORATION RESPONSE ERROR ---")
        log("AI returned invalid JSON.")
        return null
      }

      const evidenceFound = Array.isArray(parsed.evidence_found)
        ? parsed.evidence_found
        : []
      const evidenceMissing = Array.isArray(parsed.evidence_missing)
        ? parsed.evidence_missing
        : []

      for (const item of evidenceFound) {
        evidence.push({
          ...item,
          observed_at_step: step,
        })
      }

      log("--- EVIDENCE FOUND ---")
      log(JSON.stringify(evidenceFound, null, 2))

      log("--- EVIDENCE MISSING ---")
      log(JSON.stringify(evidenceMissing, null, 2))

      hooks.onEvidence?.({
        step,
        found: evidenceFound,
        missing: evidenceMissing,
      })

      log(`--- FIRSTUSER DECISION: STEP ${step} ---`)
      log(
        JSON.stringify(
          {
            action: parsed.action,
            target_index: parsed.target_index,
            scroll_direction: parsed.scroll_direction,
            scroll_amount: parsed.scroll_amount,
            reason: parsed.reason,
          },
          null,
          2
        )
      )

      return parsed
    }

    let budgetExceeded = false

    log("FIRSTUSER SESSION STARTED")

    // -----------------------------------
    // MULTI-STEP USER JOURNEY
    // -----------------------------------

    let observation = await observe()

    stepLoop: for (let step = 1; step <= MAX_STEPS; step++) {
      log(`--- STEP ${step}: OBSERVING ---`)

      observations.push(observation)

      const screenshotPath = await saveScreenshot(
        page,
        `${String(step).padStart(2, "0")}-observation.png`
      )

      hooks.onObservation?.({
        step,
        url: observation.url,
        title: observation.title,
        screenshotPath,
      })

      if (explorationBudgetReached()) {
        logExplorationBudgetReached()
        budgetExceeded = true
        break
      }

      // -----------------------------------
      // EXTRACT EVIDENCE + ASK AI WHAT TO DO NEXT (single combined call)
      // -----------------------------------

      let result

      try {
        result = await exploreAndReason(observation, step)
      } catch (error) {
        if (error instanceof LlmBudgetExceededError) {
          log("--- LLM BUDGET EXCEEDED ---")
          log(error.message)
          budgetExceeded = true
          break
        }

        throw error
      }

      if (!result) {
        log("--- ERROR ---")
        log("AI returned invalid JSON. Stopping session.")
        break
      }

      // -----------------------------------
      // STOP IF AI DECIDES NO ACTION
      // -----------------------------------

      if (result.action === "none") {
        log("--- FIRSTUSER STOPPED ---")
        log(`Reason: ${result.reason}`)
        hooks.onStopped?.(result.reason)
        break
      }

      // -----------------------------------
      // VALIDATE TARGET EXISTS
      // -----------------------------------

      const validScroll =
        result.action === "scroll" &&
        (result.scroll_direction === "down" || result.scroll_direction === "up") &&
        typeof result.scroll_amount === "number" &&
        result.scroll_amount >= 300 &&
        result.scroll_amount <= 900

      const validAction =
        (result.action === "click_button" && observation.buttons[result.target_index]) ||
        (result.action === "click_link" && observation.links[result.target_index]) ||
        (result.action === "toggle_expandable" &&
          observation.expandables[result.target_index]) ||
        validScroll

      if (!validAction) {
        log("--- INVALID ACTION ---")
        log("AI selected an invalid action. Stopping session.")
        break
      }

      // -----------------------------------
      // BLOCK REPEATED TARGETS
      // -----------------------------------

      const target =
        result.action === "click_button"
          ? observation.buttons[result.target_index]
          : result.action === "click_link"
          ? observation.links[result.target_index]
          : result.action === "toggle_expandable"
          ? observation.expandables[result.target_index]
          : null

      if (
        target &&
        (result.action === "click_button" ||
          result.action === "click_link" ||
          result.action === "toggle_expandable")
      ) {
        const alreadyVisited = explorationMemory.visitedTargets.some((item) => {
          if (result.action === "click_link") {
            return item.action === "click_link" && item.href === (target.href ?? null)
          }

          return (
            item.action === result.action &&
            item.url === observation.url &&
            item.dom_index === target.dom_index &&
            item.text === target.text
          )
        })

        if (alreadyVisited) {
          log("--- REPEATED TARGET BLOCKED ---")
          log(`FirstUser already explored: "${target.text}"`)
          break
        }
      }

      // -----------------------------------
      // EXECUTE ACTION
      // -----------------------------------

      try {
        let actionBlocked = false

        const clickSelector =
          result.action === "click_button"
            ? "button"
            : result.action === "click_link"
            ? "a"
            : result.action === "toggle_expandable"
            ? "summary, button[aria-expanded], button[aria-controls]"
            : null

        if (target && clickSelector) {
          const clickable = await isTargetClickable(clickSelector, target.dom_index)

          if (!clickable) {
            actionBlocked = true
            log("--- TARGET OCCLUDED ---")
            log(
              `FirstUser could not click "${target.text}" — it appears to be covered by another element (e.g. a blocking overlay), not the actual page content.`
            )
          }
        }

        if (result.action === "click_button" && target && !actionBlocked) {
          const button = page.locator("button").nth(target.dom_index)
          await button.click()
          log(`Clicked button [${result.target_index}] DOM [${target.dom_index}]: "${target.text}"`)
        }

        if (result.action === "click_link" && target && !actionBlocked) {
          const link = page.locator("a").nth(target.dom_index)
          log(`Attempting to click link [${result.target_index}] DOM [${target.dom_index}]: "${target.text}"`)
          await link.click()
          log(`Clicked link [${result.target_index}] DOM [${target.dom_index}]: "${target.text}"`)
        }

        if (result.action === "toggle_expandable" && target && !actionBlocked) {
          const expandable = page
            .locator("summary, button[aria-expanded], button[aria-controls]")
            .nth(target.dom_index)
          await expandable.click()
          log(`Toggled expandable [${result.target_index}] DOM [${target.dom_index}]: "${target.text}"`)
        }

        if (result.action === "scroll") {
          const scrollAmount =
            result.scroll_direction === "up" ? -result.scroll_amount : result.scroll_amount

          await page.evaluate((amount) => {
            window.scrollBy({
              top: amount,
              behavior: "smooth",
            })
          }, scrollAmount)

          log(`Scrolled ${result.scroll_direction} by ${result.scroll_amount}px.`)
        }

        await page.waitForTimeout(1500)

        // -----------------------------------
        // UPDATE EXPLORATION MEMORY
        // (skipped when blocked — nothing was actually explored, so a
        // later retry once the obstruction clears should stay available)
        // -----------------------------------

        if (
          !actionBlocked &&
          target &&
          (result.action === "click_button" ||
            result.action === "click_link" ||
            result.action === "toggle_expandable")
        ) {
          explorationMemory.visitedTargets.push({
            action: result.action,
            index: result.target_index,
            dom_index: target.dom_index,
            text: target.text,
            href: target.href ?? null,
            url: observation.url,
            step,
          })
        }

        if (result.action === "scroll") {
          explorationMemory.exploredScrollPositions.push(
            await page.evaluate(() => window.scrollY)
          )
        }

        rememberUrl(page.url())
        hooks.onNavigation?.(page.url())

        // -----------------------------------
        // SAVE ACTION HISTORY
        // (recorded whether or not the click was blocked, so a blocked
        // attempt is still visible in the journey/report)
        // -----------------------------------

        const actionRecord: ActionRecord = {
          step,
          action: result.action,
          target_index: result.target_index,
          dom_index: target?.dom_index ?? null,
          target_text: target?.text ?? null,
          target_href: result.action === "click_link" ? target?.href ?? null : null,
          reason: actionBlocked
            ? `${result.reason} (blocked: this target was covered by another element and could not actually be clicked)`
            : result.reason,
          url_after_action: page.url(),
        }

        actionHistory.push(actionRecord)
        hooks.onAction?.(actionRecord)

        // -----------------------------------
        // OBSERVE RESULT OF ACTION
        // (recorded and evidence-extracted at the top of the next iteration)
        // -----------------------------------

        observation = await observe()
      } catch (error) {
        log("--- ACTION EXECUTION ERROR ---")
        log(String(error))
        break
      }
    } // closes stepLoop

    // -----------------------------------
    // FINAL DECISION
    // -----------------------------------

    let finalDecision: FinalDecision | null
    let finalEvaluationIncomplete = false

    hooks.onEvaluating?.()

    try {
      finalDecision = await evaluateJourney(observations, actionHistory)
    } catch (error) {
      if (!(error instanceof LlmBudgetExceededError)) {
        throw error
      }

      log("--- FINAL EVALUATION INCOMPLETE ---")
      log(error.message)
      hooks.onBudgetWarning?.(
        "LLM budget was exhausted before the final evaluation could complete."
      )

      budgetExceeded = true
      finalEvaluationIncomplete = true

      finalDecision = {
        decision: "incomplete",
        reason: "LLM budget was exhausted before the final evaluation could complete.",
      }
    }

    // -----------------------------------
    // BUILD COMPLETE SESSION RESULT
    // -----------------------------------

    const sessionResult: FirstUserResult = {
      website: testConfig.websiteUrl,
      persona: testConfig.persona.trim(),
      goal: testConfig.goal.trim(),
      journey: actionHistory,
      evidence,
      decision: finalDecision,
      stoppedForLlmBudget: budgetExceeded,
      llmUsage: {
        calls: llmCallCount,
        maxCalls: MAX_LLM_CALLS,
      },
    }

    log("FIRSTUSER SESSION COMPLETE")
    log(`LLM usage: ${llmCallCount} / ${MAX_LLM_CALLS} calls`)

    if (finalEvaluationIncomplete) {
      log("Final evaluation did not complete: LLM budget exhausted.")
    } else {
      log("Exploration completed within budget.")
      log("Final evaluation completed successfully.")
    }

    hooks.onCompleted?.(sessionResult)

    return sessionResult
  } catch (error) {
    hooks.onError?.(error as Error)
    throw error
  } finally {
    await browser.close()
  }
}
