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

export interface FirstUserConfig {
  websiteUrl: string
  persona: string
  goal: string
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

  const screenshotDir = options.screenshotDir ?? "screenshots"

  const log = (message: string) => hooks.onLog?.(message)

  let llmCallCount = 0

  // One call is always reserved for evaluateJourney(), so exploration
  // (evidence extraction + action reasoning) may use at most this many.
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
              }
            })
            .filter((item) => item.visible)
            .map((item, visibleIndex) => ({
              index: visibleIndex,
              dom_index: item.dom_index,
              text: item.text,
              visible: item.visible,
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
              }
            })
            .filter((item) => item.visible)
            .map((item, visibleIndex) => ({
              index: visibleIndex,
              dom_index: item.dom_index,
              text: item.text,
              href: item.href,
              visible: item.visible,
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

      return {
        url: page.url(),
        title: await page.title(),
        viewport,

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

        buttons,
        links,
        expandables,
        visibleText: visibleText.slice(0, 4000),
        pageTextLength: visibleText.length,
        hasForm: formFieldCount > 0,
      }
    }

    // -----------------------------------
    // EXTRACT RELEVANT EVIDENCE
    // -----------------------------------

    async function extractEvidence(observation: any, step: number) {
      log(`--- EXTRACTING EVIDENCE: STEP ${step} ---`)

      const result = await callOpenAI("evidence extraction", {
        model: "gpt-5",

        input: `
You are the evidence extraction layer of FirstUser.

An AI visitor is exploring a website with a specific goal.

Your job is NOT to make the final decision.

Your job is to identify only the pieces of information in the
CURRENT VIEWPORT that materially help answer the visitor's goal.

WEBSITE UNDER TEST:
${testConfig.websiteUrl}

VISITOR PERSONA:
${testConfig.persona}

VISITOR GOAL:
${testConfig.goal}

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

Ask yourself:

"Could the visitor reasonably accomplish their stated goal without
this information?"

If yes, do NOT classify it as critical missing evidence.

CURRENT STEP:
${step}

CURRENT VIEWPORT OBSERVATION:

${JSON.stringify(observation, null, 2)}

EVIDENCE ALREADY COLLECTED FROM EARLIER STEPS:

${JSON.stringify(evidence, null, 2)}

Return ONLY valid JSON:

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
  ]
}

If there is no relevant evidence, return empty arrays.
`,
      })

      let extracted

      try {
        extracted = JSON.parse(result.output_text)
      } catch (error) {
        log("--- EVIDENCE EXTRACTION ERROR ---")
        log("AI returned invalid JSON.")
        log(result.output_text)
        return
      }

      for (const item of extracted.evidence_found ?? []) {
        evidence.push({
          ...item,
          observed_at_step: step,
        })
      }

      log("--- EVIDENCE FOUND ---")
      log(JSON.stringify(extracted.evidence_found ?? [], null, 2))

      log("--- EVIDENCE MISSING ---")
      log(JSON.stringify(extracted.evidence_missing ?? [], null, 2))

      hooks.onEvidence?.({
        step,
        found: extracted.evidence_found ?? [],
        missing: extracted.evidence_missing ?? [],
      })
    }

    // -----------------------------------
    // KEEP TRACK OF USER JOURNEY
    // -----------------------------------

    const actionHistory: ActionRecord[] = []
    const observations: any[] = []
    let observationRecorded = false
    let budgetExceeded = false

    log("FIRSTUSER SESSION STARTED")

    // -----------------------------------
    // MULTI-STEP USER JOURNEY
    // -----------------------------------

    let observation = await observe()

    stepLoop: for (let step = 1; step <= MAX_STEPS; step++) {
      if (!observationRecorded) {
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

        try {
          await extractEvidence(observation, step)
        } catch (error) {
          if (error instanceof LlmBudgetExceededError) {
            log("--- LLM BUDGET EXCEEDED ---")
            log(error.message)
            budgetExceeded = true
            break
          }

          throw error
        }

        observationRecorded = true
      }

      // -----------------------------------
      // ASK AI WHAT TO DO NEXT
      // -----------------------------------

      if (explorationBudgetReached()) {
        logExplorationBudgetReached()
        budgetExceeded = true
        break
      }

      let decision

      try {
        decision = await callOpenAI("action reasoning", {
          model: "gpt-5",

          input: `
You are FirstUser, an AI agent behaving like a thoughtful first-time user exploring a website.

WEBSITE UNDER TEST:
${testConfig.websiteUrl}

VISITOR PERSONA:
${testConfig.persona}

VISITOR GOAL:
${testConfig.goal}

Your job is to explore the website naturally and determine whether you can accomplish this goal.

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

1. What useful information is already visible in the current viewport?
2. Is there likely to be meaningful information further down this page?
3. If so, is scrolling the best next action?
4. If not, is there a useful visible interaction (link, button, expandable)?
5. If the current page is no longer likely to produce useful evidence,
   navigate to another page.

This is a guide for reasoning, not a fixed sequence. If a navigation
destination (e.g. "Pricing") is clearly the fastest way to answer the
visitor's goal, choose it immediately — do not scroll first merely for
the sake of scrolling.

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

EVIDENCE DISCOVERED SO FAR:

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

CURRENT PAGE OBSERVATION:

${JSON.stringify(observation, null, 2)}

Buttons, links, and expandable elements are numbered by index.

Some elements may have the same visible text.

For example:

- Link index 2: "How it works"
- Link index 9: "How it works"

These may be different elements.

Choose the exact index you want.

Choose ONE action.

The action must be one of:

- click_button
- click_link
- toggle_expandable
- scroll
- none

Only choose an index that actually exists for that action type.

Return JSON only:

{
  "action": "click_button" | "click_link" | "toggle_expandable" | "scroll" | "none",
  "target_index": number | null,
  "scroll_direction": "down" | "up" | null,
  "scroll_amount": number | null,
  "reason": "why this is the most useful next action"
}
`,
        })
      } catch (error) {
        if (error instanceof LlmBudgetExceededError) {
          log("--- LLM BUDGET EXCEEDED ---")
          log(error.message)
          budgetExceeded = true
          break
        }

        throw error
      }

      log(`--- FIRSTUSER DECISION: STEP ${step} ---`)
      log(decision.output_text)

      let result

      try {
        result = JSON.parse(decision.output_text)
      } catch (error) {
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
        if (result.action === "click_button" && target) {
          const button = page.locator("button").nth(target.dom_index)
          await button.click()
          log(`Clicked button [${result.target_index}] DOM [${target.dom_index}]: "${target.text}"`)
        }

        if (result.action === "click_link" && target) {
          const link = page.locator("a").nth(target.dom_index)
          log(`Attempting to click link [${result.target_index}] DOM [${target.dom_index}]: "${target.text}"`)
          await link.click()
          log(`Clicked link [${result.target_index}] DOM [${target.dom_index}]: "${target.text}"`)
        }

        if (result.action === "toggle_expandable" && target) {
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
        // -----------------------------------

        if (
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
        // -----------------------------------

        const actionRecord: ActionRecord = {
          step,
          action: result.action,
          target_index: result.target_index,
          dom_index: target?.dom_index ?? null,
          target_text: target?.text ?? null,
          target_href: result.action === "click_link" ? target?.href ?? null : null,
          reason: result.reason,
          url_after_action: page.url(),
        }

        actionHistory.push(actionRecord)
        hooks.onAction?.(actionRecord)

        // -----------------------------------
        // OBSERVE RESULT OF ACTION
        // -----------------------------------

        observation = await observe()
        observationRecorded = false

        observations.push(observation)

        const screenshotPath = await saveScreenshot(
          page,
          `${String(step + 1).padStart(2, "0")}-observation.png`
        )

        hooks.onObservation?.({
          step: step + 1,
          url: observation.url,
          title: observation.title,
          screenshotPath,
        })

        if (explorationBudgetReached()) {
          logExplorationBudgetReached()
          budgetExceeded = true
          break
        }

        await extractEvidence(observation, step + 1)

        observationRecorded = true
      } catch (error) {
        if (error instanceof LlmBudgetExceededError) {
          log("--- LLM BUDGET EXCEEDED ---")
          log((error as Error).message)
          budgetExceeded = true
          break
        }

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
