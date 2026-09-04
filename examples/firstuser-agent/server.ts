/**
 * FirstUser API server.
 *
 * Thin HTTP layer over the existing agent (agent.ts). Starts a real
 * FirstUser session per request, tracks its progress in memory, and
 * exposes only safe, user-facing state to the frontend — never API keys,
 * prompts, or raw model output.
 *
 * V1 on purpose: in-memory session store, no auth, no database, no queue.
 */

import "dotenv/config"
import express from "express"
import { randomUUID } from "node:crypto"
import { existsSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import {
  closeSolariClient,
  runFirstUserTest,
  WebsiteUnreachableError,
  LlmBudgetExceededError,
  type FirstUserResult,
  type ResearchRules,
} from "./agent.ts"
import { renderReportPdf, type ReportPdfInput } from "./pdf-report.ts"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const FRONTEND_DIR = path.join(__dirname, "frontend")
const SCREENSHOTS_ROOT = path.join(__dirname, "screenshots")

const PORT = Number(process.env.PORT) || 3000

type TestStatus = "starting" | "exploring" | "evaluating" | "completed" | "error"

interface TestSession {
  id: string
  status: TestStatus
  websiteUrl: string
  persona: string
  goal: string
  createdAt: Date
  currentUrl: string | null
  currentActivity: string
  observationCount: number
  actionCount: number
  evidenceCount: number
  journeyLog: string[]
  screenshotFiles: string[]
  result: FirstUserResult | null
  error: string | null
}

const sessions = new Map<string, TestSession>()

const MAX_JOURNEY_LOG = 100

function pushLog(session: TestSession, message: string) {
  session.journeyLog.push(message)
  if (session.journeyLog.length > MAX_JOURNEY_LOG) {
    session.journeyLog.shift()
  }
}

function truncate(text: string, max: number) {
  const trimmed = text.trim()
  return trimmed.length > max ? `${trimmed.slice(0, max - 1)}…` : trimmed
}

function describeAction(action: string, targetText: string | null) {
  switch (action) {
    case "click_link":
    case "click_button":
      return targetText ? `Opened "${targetText}"` : "Opened a linked section"
    case "toggle_expandable":
      return targetText ? `Expanded "${targetText}"` : "Expanded a section"
    case "scroll":
      return "Scrolled to explore more of the page"
    default:
      return "Took an exploration step"
  }
}

/** Maps an internal error to a calm, user-facing message. Never leaks stack traces or key material. */
function describeError(error: unknown): string {
  if (error instanceof WebsiteUnreachableError) {
    return "The website could not be reached. Check the URL and try again."
  }
  if (error instanceof LlmBudgetExceededError) {
    return "FirstUser's exploration budget was used up before it could finish."
  }
  return "FirstUser ran into an unexpected problem during this session."
}

function isValidHttpUrl(value: string): boolean {
  try {
    const url = new URL(value)
    return url.protocol === "http:" || url.protocol === "https:"
  } catch {
    return false
  }
}

/** Accepts loose/partial input from the request body; unrecognized or non-boolean values fall back to the safe default (`true`) inside runFirstUserTest. */
function sanitizeResearchRules(value: unknown): ResearchRules {
  const input = (value ?? {}) as Record<string, unknown>
  const rules: ResearchRules = {}

  if (typeof input.evidenceOnly === "boolean") rules.evidenceOnly = input.evidenceOnly
  if (typeof input.exploreBeforeDeciding === "boolean") rules.exploreBeforeDeciding = input.exploreBeforeDeciding
  if (typeof input.allowUncertain === "boolean") rules.allowUncertain = input.allowUncertain

  return rules
}

function startTest(
  websiteUrl: string,
  persona: string,
  goal: string,
  researchRules: ResearchRules
): TestSession {
  const id = randomUUID()

  const session: TestSession = {
    id,
    status: "starting",
    websiteUrl,
    persona,
    goal,
    createdAt: new Date(),
    currentUrl: null,
    currentActivity: "Starting the browser session",
    observationCount: 0,
    actionCount: 0,
    evidenceCount: 0,
    journeyLog: [],
    screenshotFiles: [],
    result: null,
    error: null,
  }

  sessions.set(id, session)

  runFirstUserTest(
    { websiteUrl, persona, goal, researchRules },
    {
      onSessionStarted: () => {
        session.status = "exploring"
        session.currentUrl = websiteUrl
        session.currentActivity = "Opening the homepage"
        pushLog(session, "Opened homepage")
      },
      onNavigation: (url) => {
        session.currentUrl = url
      },
      onObservation: ({ step, url, screenshotPath }) => {
        session.observationCount = step
        session.currentUrl = url
        session.screenshotFiles.push(path.basename(screenshotPath))
      },
      onEvidence: ({ found }) => {
        if (found.length > 0) {
          session.evidenceCount += found.length
          const label = `Evidence captured: ${truncate(found[0].claim, 90)}`
          session.currentActivity = label
          pushLog(session, label)
        }
      },
      onAction: (record) => {
        session.actionCount += 1
        session.currentUrl = record.url_after_action
        const label = describeAction(record.action, record.target_text)
        session.currentActivity = label
        pushLog(session, label)
      },
      onStopped: (reason) => {
        pushLog(session, `Stopped exploring: ${truncate(reason, 120)}`)
      },
      onEvaluating: () => {
        session.status = "evaluating"
        session.currentActivity = "Compiling the final verdict"
        pushLog(session, "Compiling final verdict")
      },
      onBudgetWarning: (message) => {
        pushLog(session, message)
      },
      onCompleted: (result) => {
        session.status = "completed"
        session.result = result
        session.currentActivity = "Investigation complete"
        pushLog(session, "Investigation complete")
      },
      onError: (error) => {
        session.status = "error"
        session.error = describeError(error)
        session.currentActivity = "Investigation failed"
        pushLog(session, `Investigation failed: ${session.error}`)
      },
    },
    { screenshotDir: path.join("screenshots", id) }
  ).catch(() => {
    // onError hook above already recorded the safe, user-facing state.
    // Nothing further to do here — this catch only stops an unhandled
    // rejection from surfacing.
  })

  return session
}

function toStatusPayload(session: TestSession) {
  return {
    status: session.status,
    currentUrl: session.currentUrl,
    currentActivity: session.currentActivity,
    observations: session.observationCount,
    actions: session.actionCount,
    evidenceCount: session.evidenceCount,
    journeyLog: session.journeyLog,
    screenshots: session.screenshotFiles.map(
      (file) => `/api/tests/${session.id}/screenshots/${file}`
    ),
    error: session.error,
  }
}

function toResultPayload(session: TestSession) {
  if (session.status === "error") {
    return {
      status: "error",
      website: session.websiteUrl,
      persona: session.persona,
      goal: session.goal,
      error: session.error,
    }
  }

  const result = session.result
  const decision = result?.decision ?? null

  return {
    status: session.status,
    website: result?.website ?? session.websiteUrl,
    persona: result?.persona ?? session.persona,
    goal: result?.goal ?? session.goal,
    decision: decision?.decision ?? "uncertain",
    confidence: decision?.confidence ?? null,
    reason: decision?.reason ?? null,
    recommendation: decision?.recommendation ?? null,
    evidence_found: decision?.evidence_found ?? [],
    evidence_missing: decision?.evidence_missing ?? [],
    what_helped: decision?.what_helped ?? [],
    what_blocked: decision?.what_blocked ?? [],
    evidence_trail: result?.evidence ?? [],
    journey: result?.journey ?? [],
    screenshots: session.screenshotFiles.map(
      (file) => `/api/tests/${session.id}/screenshots/${file}`
    ),
    stoppedForLlmBudget: result?.stoppedForLlmBudget ?? false,
  }
}

/**
 * Builds the PDF renderer's input straight from the same session/result
 * that powers toResultPayload() above, so the downloaded report and the
 * web Report screen are always describing the same investigation.
 *
 * Returns null if the session has no completed decision yet — callers
 * must treat that as "not ready to export," not as an empty report.
 */
function buildReportPdfInput(session: TestSession): ReportPdfInput | null {
  const decision = session.result?.decision
  if (session.status !== "completed" || !session.result || !decision) return null

  const rawScreenshots = session.screenshotFiles

  const evidence = session.result.evidence.map((item) => {
    const file = rawScreenshots[item.observed_at_step - 1]
    const screenshotPath = file ? path.join(SCREENSHOTS_ROOT, session.id, file) : null
    return {
      claim: item.claim,
      why: item.why_it_matters,
      source: item.source,
      screenshotPath: screenshotPath && existsSync(screenshotPath) ? screenshotPath : null,
    }
  })

  return {
    website: session.websiteUrl,
    persona: session.persona,
    goal: session.goal,
    investigatedAt: session.createdAt,
    decisionKey: decision.decision ?? "uncertain",
    confidenceKey: decision.confidence ?? null,
    reason: decision.reason ?? "",
    helped: decision.what_helped ?? [],
    blocked: decision.what_blocked ?? [],
    recommendation: decision.recommendation ?? null,
    evidence,
    evidenceGaps: (decision.evidence_missing ?? []).map((claim) => ({ claim })),
    journey: session.result.journey.map((rec) => ({
      num: String(rec.step).padStart(2, "0"),
      title: describeAction(rec.action, rec.target_text),
      desc: rec.reason,
    })),
  }
}

function sanitizeForFilename(value: string): string {
  return value
    .toLowerCase()
    .replace(/^https?:\/\//, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
}

function buildReportFilename(websiteUrl: string, date: Date): string {
  const domain = sanitizeForFilename(websiteUrl) || "website"
  const iso = date.toISOString().slice(0, 10)
  return `firstuser-report-${domain}-${iso}.pdf`
}

const app = express()
app.use(express.json())

app.post("/api/tests", (req, res) => {
  const { websiteUrl, persona, goal, researchRules } = req.body ?? {}

  if (typeof websiteUrl !== "string" || !isValidHttpUrl(websiteUrl)) {
    res.status(400).json({ error: "Enter a valid http:// or https:// website URL." })
    return
  }

  if (typeof persona !== "string" || !persona.trim()) {
    res.status(400).json({ error: "Persona is required." })
    return
  }

  if (typeof goal !== "string" || !goal.trim()) {
    res.status(400).json({ error: "Goal is required." })
    return
  }

  const session = startTest(
    websiteUrl.trim(),
    persona.trim(),
    goal.trim(),
    sanitizeResearchRules(researchRules)
  )

  res.status(202).json({ testId: session.id, status: session.status })
})

app.get("/api/tests/:id/status", (req, res) => {
  const session = sessions.get(req.params.id)

  if (!session) {
    res.status(404).json({ error: "Test not found." })
    return
  }

  res.json(toStatusPayload(session))
})

app.get("/api/tests/:id/result", (req, res) => {
  const session = sessions.get(req.params.id)

  if (!session) {
    res.status(404).json({ error: "Test not found." })
    return
  }

  if (session.status !== "completed" && session.status !== "error") {
    res.status(409).json({ error: "Test is still running.", status: session.status })
    return
  }

  res.json(toResultPayload(session))
})

app.get("/api/tests/:id/report.pdf", async (req, res) => {
  const session = sessions.get(req.params.id)

  if (!session) {
    res.status(404).json({ error: "Test not found." })
    return
  }

  const pdfInput = buildReportPdfInput(session)

  if (!pdfInput) {
    res.status(409).json({ error: "This investigation hasn't completed yet." })
    return
  }

  try {
    const pdf = await renderReportPdf(pdfInput)
    const filename = buildReportFilename(session.websiteUrl, session.createdAt)
    res.setHeader("Content-Type", "application/pdf")
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`)
    res.send(pdf)
  } catch (err) {
    console.error(err)
    res.status(500).json({ error: "Could not generate the PDF report." })
  }
})

const SCREENSHOT_FILE_RE = /^\d{2}-observation\.png$/

app.get("/api/tests/:id/screenshots/:file", (req, res) => {
  const session = sessions.get(req.params.id)

  if (!session) {
    res.status(404).json({ error: "Test not found." })
    return
  }

  if (!SCREENSHOT_FILE_RE.test(req.params.file)) {
    res.status(400).json({ error: "Invalid screenshot name." })
    return
  }

  const filePath = path.join(SCREENSHOTS_ROOT, session.id, req.params.file)

  if (!existsSync(filePath)) {
    res.status(404).json({ error: "Screenshot not available yet." })
    return
  }

  res.sendFile(filePath)
})

app.use(express.static(FRONTEND_DIR))

app.get("/", (_req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, "FirstUser.dc.html"))
})

app.use((err: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error(err)
  res.status(500).json({ error: "Unexpected server error." })
})

const server = app.listen(PORT, () => {
  console.log(`FirstUser server listening on http://localhost:${PORT}`)
})

async function shutdown() {
  server.close()
  await closeSolariClient()
  process.exit(0)
}

process.on("SIGINT", shutdown)
process.on("SIGTERM", shutdown)
