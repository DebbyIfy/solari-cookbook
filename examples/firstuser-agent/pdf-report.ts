/**
 * FirstUser PDF report renderer.
 *
 * Turns a completed, LIVE investigation into a paginated, print-quality PDF
 * that reads like a research document — not a webpage screenshot. This
 * module owns layout only: every fact it draws (verdict, evidence, journey,
 * screenshots) is handed in already derived from the same session/decision
 * data that powers the web Report screen (see server.ts), so the two views
 * cannot drift apart.
 *
 * Integrity rule enforced here: a screenshot is only ever drawn when the
 * caller supplies a real, on-disk `screenshotPath` for that evidence item.
 * There is no fallback, placeholder, or generated image anywhere below.
 */

import { closeSync, openSync, readSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import PDFDocument from "pdfkit"

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// DejaVu ships full-coverage Unicode glyphs (Bitstream Vera-derived, freely
// redistributable) instead of PDFKit's 14 standard fonts, whose WinAnsi
// encoding drops glyphs report content commonly needs — e.g. the Naira
// sign (₦) that shows up in FirstUser's own example mission text.
const FONT_DIR = path.join(__dirname, "node_modules", "dejavu-fonts-ttf", "ttf")
const FONT_FILES = {
  Sans: "DejaVuSans.ttf",
  SansBold: "DejaVuSans-Bold.ttf",
  Serif: "DejaVuSerif.ttf",
  SerifItalic: "DejaVuSerif-Italic.ttf",
  Mono: "DejaVuSansMono.ttf",
  MonoBold: "DejaVuSansMono-Bold.ttf",
} as const

function registerFonts(doc: PDFKit.PDFDocument) {
  for (const [name, file] of Object.entries(FONT_FILES)) {
    doc.registerFont(name, path.join(FONT_DIR, file))
  }
}

export interface PdfEvidenceItem {
  claim: string
  why: string
  source: string
  /** Absolute path to a real screenshot captured for this evidence item, or null if none was matched. */
  screenshotPath: string | null
}

export interface PdfEvidenceGap {
  claim: string
}

export interface PdfJourneyStep {
  num: string
  title: string
  desc: string
}

export interface ReportPdfInput {
  website: string
  persona: string
  goal: string
  investigatedAt: Date
  decisionKey: "yes" | "no" | "uncertain" | "incomplete"
  confidenceKey: "low" | "medium" | "high" | null
  reason: string
  helped: string[]
  blocked: string[]
  recommendation: string | null
  evidence: PdfEvidenceItem[]
  evidenceGaps: PdfEvidenceGap[]
  journey: PdfJourneyStep[]
}

// ---- Palette — matches the web Report card (frontend/FirstUser.dc.html) ----

const INK = "#1B1B18"
const MUTED = "#6E7178"
const LABEL = "#8C897E"
const BORDER = "#DCD8CC"
const HELPED_DOT = "#7BA7FF"
const BLOCKED_DOT = "#E87575"
const GAP_LABEL = "#C24444"
const RECOMMENDATION_BG = "#EAE7DF"

const DECISION_STYLES: Record<ReportPdfInput["decisionKey"], { label: string; color: string }> = {
  yes: { label: "Yes", color: "#3E7A34" },
  no: { label: "No", color: "#B23A3A" },
  uncertain: { label: "Uncertain", color: "#B8791F" },
  incomplete: { label: "Incomplete", color: "#6E7178" },
}

const CONFIDENCE_STYLES: Record<"low" | "medium" | "high", { label: string; color: string }> = {
  high: { label: "High confidence", color: "#D99B2B" },
  medium: { label: "Medium confidence", color: "#4C7EC9" },
  low: { label: "Low confidence", color: "#8C897E" },
}

const UNCERTAINTY_NOTE =
  "FirstUser found enough evidence to confidently conclude that the information available on the website is insufficient to answer this question with certainty."

const MARGIN = 56
const MAX_IMAGE_HEIGHT = 260

export async function renderReportPdf(input: ReportPdfInput): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({
      size: "A4",
      margins: { top: MARGIN, bottom: 64, left: MARGIN, right: MARGIN },
      bufferPages: true,
      info: {
        Title: `FirstUser Report — ${input.website}`,
        Author: "FirstUser",
        Subject: "FirstUser investigation report",
      },
    })

    const chunks: Buffer[] = []
    doc.on("data", (chunk) => chunks.push(chunk))
    doc.on("end", () => resolve(Buffer.concat(chunks)))
    doc.on("error", reject)

    try {
      registerFonts(doc)
      draw(doc, input)
      stampFooters(doc)
      doc.end()
    } catch (err) {
      reject(err instanceof Error ? err : new Error(String(err)))
    }
  })
}

function contentWidth(doc: PDFKit.PDFDocument): number {
  return doc.page.width - doc.page.margins.left - doc.page.margins.right
}

/** Starts a new page if `needed` more points won't fit above the footer margin. */
function ensureSpace(doc: PDFKit.PDFDocument, needed: number) {
  const bottom = doc.page.height - doc.page.margins.bottom
  if (doc.y + needed > bottom) doc.addPage()
}

function resetX(doc: PDFKit.PDFDocument) {
  doc.x = doc.page.margins.left
}

function sectionLabel(doc: PDFKit.PDFDocument, text: string, color: string = LABEL) {
  ensureSpace(doc, 30)
  doc.font("SansBold").fontSize(10).fillColor(color).text(text.toUpperCase(), { characterSpacing: 0.6 })
  doc.moveDown(0.7)
  resetX(doc)
}

function draw(doc: PDFKit.PDFDocument, input: ReportPdfInput) {
  drawCover(doc, input)

  doc.addPage()
  drawVerdict(doc, input)

  ensureSpace(doc, 90)
  const columnsStartY = doc.y
  const colWidth = (contentWidth(doc) - 24) / 2
  const helpedBottom = drawBulletColumn(
    doc,
    doc.page.margins.left,
    columnsStartY,
    colWidth,
    "What helped",
    input.helped,
    HELPED_DOT,
    false
  )
  const rightX = doc.page.margins.left + colWidth + 24
  const blockedBottom = drawBulletColumn(
    doc,
    rightX,
    columnsStartY,
    colWidth,
    "What blocked the decision",
    input.blocked,
    BLOCKED_DOT,
    true
  )
  doc.y = Math.max(helpedBottom, blockedBottom) + 22
  resetX(doc)

  if (input.recommendation) {
    drawRecommendation(doc, input.recommendation)
  }

  doc.addPage()
  sectionLabel(doc, "Evidence trail")
  input.evidence.forEach((item, i) => drawEvidenceCard(doc, item, i + 1))

  if (input.evidenceGaps.length > 0) {
    doc.moveDown(0.4)
    sectionLabel(doc, "Evidence gaps", GAP_LABEL)
    input.evidenceGaps.forEach((gap) => drawGapRow(doc, gap))
  }

  if (input.journey.length > 0) {
    doc.moveDown(0.8)
    sectionLabel(doc, "Research journey")
    input.journey.forEach((step) => drawJourneyRow(doc, step))
  }
}

// -----------------------------------------------------------------
// Cover
// -----------------------------------------------------------------

function drawCover(doc: PDFKit.PDFDocument, input: ReportPdfInput) {
  const width = contentWidth(doc)

  doc.font("SansBold").fontSize(13).fillColor(INK).text("FIRSTUSER", { characterSpacing: 3 })
  doc.font("SerifItalic").fontSize(16).fillColor(MUTED).text("Research Report")
  doc.moveDown(1.6)

  doc
    .moveTo(doc.page.margins.left, doc.y)
    .lineTo(doc.page.margins.left + width, doc.y)
    .lineWidth(1)
    .strokeColor(BORDER)
    .stroke()
  doc.moveDown(1.6)

  sectionLabel(doc, "Target")
  doc.font("Mono").fontSize(15).fillColor(INK).text(stripProtocol(input.website))
  doc.moveDown(1.1)

  sectionLabel(doc, "Mission")
  doc
    .font("SerifItalic")
    .fontSize(13)
    .fillColor(INK)
    .text(`"${input.goal}"`, { width, lineGap: 4 })
  doc.moveDown(1.1)

  sectionLabel(doc, "Investigated as")
  doc.font("Sans").fontSize(12).fillColor(INK).text(input.persona)
  doc.moveDown(1.1)

  sectionLabel(doc, "Date of investigation")
  doc.font("Sans").fontSize(12).fillColor(INK).text(formatDate(input.investigatedAt))
  doc.moveDown(1.6)

  drawBadge(doc, "LIVE SESSION")
}

function drawBadge(doc: PDFKit.PDFDocument, text: string) {
  const paddingX = 11
  const height = 22
  doc.font("MonoBold").fontSize(9)
  const textW = doc.widthOfString(text, { characterSpacing: 1 })
  const w = textW + paddingX * 2
  const x = doc.page.margins.left
  const y = doc.y

  doc
    .lineWidth(1)
    .roundedRect(x, y, w, height, 5)
    .fillAndStroke("#EEF6D6", "#B9CC85")
  doc
    .fillColor("#4A5A26")
    .text(text, x + paddingX, y + (height - 9) / 2 - 1, { characterSpacing: 1, lineBreak: false })

  doc.y = y + height + 8
  resetX(doc)
}

// -----------------------------------------------------------------
// Verdict — Verdict -> Explanation -> Clarification -> Confidence,
// matching the semantic hierarchy on the web Report screen exactly.
// -----------------------------------------------------------------

function drawVerdict(doc: PDFKit.PDFDocument, input: ReportPdfInput) {
  const width = contentWidth(doc)
  const dStyle = DECISION_STYLES[input.decisionKey]
  const confStyle = input.confidenceKey ? CONFIDENCE_STYLES[input.confidenceKey] : null

  sectionLabel(doc, "Verdict")
  doc.font("Serif").fontSize(46).fillColor(dStyle.color).text(dStyle.label)
  doc.moveDown(0.6)

  doc.font("Sans").fontSize(13).fillColor(INK).text(input.reason, { width, lineGap: 4 })
  doc.moveDown(0.5)

  const showUncertaintyNote = input.decisionKey === "uncertain" && input.confidenceKey === "high"
  if (showUncertaintyNote) {
    doc.font("Sans").fontSize(11).fillColor(MUTED).text(UNCERTAINTY_NOTE, { width, lineGap: 3 })
    doc.moveDown(0.6)
  }

  doc.moveDown(0.3)
  const confidenceY = doc.y
  doc.circle(doc.page.margins.left + 3, confidenceY + 4, 3).fill(confStyle ? confStyle.color : LABEL)
  doc
    .font("SansBold")
    .fontSize(10)
    .fillColor(LABEL)
    .text(
      confStyle ? `${confStyle.label.toUpperCase()} IN THIS CONCLUSION` : "CONFIDENCE UNAVAILABLE",
      doc.page.margins.left + 12,
      confidenceY,
      { characterSpacing: 0.5 }
    )
  doc.moveDown(1.2)
  resetX(doc)
}

// -----------------------------------------------------------------
// What helped / What blocked
// -----------------------------------------------------------------

function drawBulletColumn(
  doc: PDFKit.PDFDocument,
  x: number,
  y: number,
  width: number,
  title: string,
  items: string[],
  dotColor: string,
  outlineDot: boolean
): number {
  doc.font("SansBold").fontSize(11.5).fillColor(INK).text(title, x, y, { width })
  let cy = doc.y + 10

  doc.font("Sans").fontSize(10)
  const textWidth = width - 16
  for (const item of items) {
    const h = doc.heightOfString(item, { width: textWidth, lineGap: 2 })
    if (outlineDot) {
      doc.lineWidth(1.3).circle(x + 3, cy + 5, 3).stroke(dotColor)
    } else {
      doc.circle(x + 3, cy + 5, 3).fill(dotColor)
    }
    doc.fillColor(INK).text(item, x + 16, cy, { width: textWidth, lineGap: 2 })
    cy = doc.y + 10
  }

  return cy
}

function drawRecommendation(doc: PDFKit.PDFDocument, text: string) {
  const width = contentWidth(doc)
  const pad = 20
  const textWidth = width - pad * 2

  doc.font("SansBold").fontSize(9)
  const labelH = 12
  doc.font("SerifItalic").fontSize(13)
  const textH = doc.heightOfString(text, { width: textWidth, lineGap: 3 })
  const boxH = pad * 2 + labelH + 8 + textH

  ensureSpace(doc, boxH + 24)
  const x = doc.page.margins.left
  const y = doc.y

  doc.roundedRect(x, y, width, boxH, 10).fill(RECOMMENDATION_BG)
  doc
    .font("SansBold")
    .fontSize(9)
    .fillColor(LABEL)
    .text("RECOMMENDATION", x + pad, y + pad, { characterSpacing: 0.6 })
  doc
    .font("SerifItalic")
    .fontSize(13)
    .fillColor(INK)
    .text(text, x + pad, y + pad + labelH + 8, { width: textWidth, lineGap: 3 })

  doc.y = y + boxH + 24
  resetX(doc)
}

// -----------------------------------------------------------------
// Evidence trail
// -----------------------------------------------------------------

function drawEvidenceCard(doc: PDFKit.PDFDocument, item: PdfEvidenceItem, index: number) {
  const width = contentWidth(doc)
  const pad = 16
  const textWidth = width - pad * 2
  const label = `EVIDENCE ${String(index).padStart(2, "0")}`

  doc.font("SansBold").fontSize(12)
  const claimH = doc.heightOfString(item.claim, { width: textWidth, lineGap: 2 })

  doc.font("Sans").fontSize(10)
  const whyText = `Why it matters: ${item.why}`
  const whyH = doc.heightOfString(whyText, { width: textWidth, lineGap: 2 })

  const sourceH = 22

  let image: { path: string; width: number; height: number } | null = null
  if (item.screenshotPath) {
    const dims = readPngDimensions(item.screenshotPath)
    if (dims) {
      const scale = Math.min(textWidth / dims.width, MAX_IMAGE_HEIGHT / dims.height, 1)
      image = { path: item.screenshotPath, width: dims.width * scale, height: dims.height * scale }
    }
  }

  const labelH = 12
  const cardH =
    pad + labelH + 8 + claimH + 10 + whyH + 14 + sourceH + pad + (image ? image.height + 16 : 0)

  ensureSpace(doc, cardH + 16)
  const x = doc.page.margins.left
  const y = doc.y

  doc.lineWidth(1).strokeColor(BORDER).roundedRect(x, y, width, cardH, 12).stroke()

  let cy = y + pad
  doc
    .font("Mono")
    .fontSize(9)
    .fillColor("#9C988C")
    .text(label, x + pad, cy, { characterSpacing: 0.6, lineBreak: false })
  cy += labelH + 8

  doc.font("SansBold").fontSize(12).fillColor(INK).text(item.claim, x + pad, cy, { width: textWidth, lineGap: 2 })
  cy += claimH + 10

  doc
    .font("SansBold")
    .fontSize(10)
    .fillColor(LABEL)
    .text("Why it matters: ", x + pad, cy, { continued: true, width: textWidth, lineGap: 2 })
  doc.font("Sans").fillColor(MUTED).text(item.why)
  cy += whyH + 14

  const tagText = `Observed on: ${item.source}`
  doc.font("Mono").fontSize(9)
  const tagW = doc.widthOfString(tagText) + 14
  doc.lineWidth(1).strokeColor(BORDER).roundedRect(x + pad, cy, tagW, 18, 4).stroke()
  doc.fillColor(MUTED).text(tagText, x + pad + 7, cy + 5, { lineBreak: false })
  cy += sourceH

  if (image) {
    const imgX = x + pad + (textWidth - image.width) / 2
    doc.image(image.path, imgX, cy, { width: image.width, height: image.height })
    cy += image.height + 16
  }

  doc.y = y + cardH + 14
  resetX(doc)
}

function drawGapRow(doc: PDFKit.PDFDocument, gap: PdfEvidenceGap) {
  const width = contentWidth(doc)
  const textWidth = width - 16
  const h = doc.font("SansBold").fontSize(10.5).heightOfString(gap.claim, { width: textWidth, lineGap: 2 })

  ensureSpace(doc, h + 14)
  const x = doc.page.margins.left
  const y = doc.y

  doc.lineWidth(1.3).circle(x + 3, y + 6, 3).stroke(BLOCKED_DOT)
  doc.font("SansBold").fontSize(10.5).fillColor(INK).text(gap.claim, x + 16, y, { width: textWidth, lineGap: 2 })

  doc.y = Math.max(doc.y, y + h) + 10
  resetX(doc)
}

// -----------------------------------------------------------------
// Research journey
// -----------------------------------------------------------------

function drawJourneyRow(doc: PDFKit.PDFDocument, step: PdfJourneyStep) {
  const width = contentWidth(doc)
  const numWidth = 32
  const textWidth = width - numWidth - 12

  doc.font("SansBold").fontSize(11)
  const titleH = doc.heightOfString(step.title, { width: textWidth })
  doc.font("Sans").fontSize(9.5)
  const descH = doc.heightOfString(step.desc, { width: textWidth, lineGap: 2 })
  const rowH = titleH + descH + 4

  ensureSpace(doc, rowH + 18)
  const x = doc.page.margins.left
  const y = doc.y

  doc.font("Serif").fontSize(17).fillColor("#B5B0A2").text(step.num, x, y, { width: numWidth, lineBreak: false })
  doc.font("SansBold").fontSize(11).fillColor(INK).text(step.title, x + numWidth + 12, y, { width: textWidth })
  doc.font("Sans").fontSize(9.5).fillColor(MUTED).text(step.desc, x + numWidth + 12, doc.y, { width: textWidth, lineGap: 2 })

  doc.y = Math.max(doc.y, y + rowH) + 16
  resetX(doc)
}

// -----------------------------------------------------------------
// Footer — stamped on every page after content is laid out, since
// the total page count isn't known until the document is complete.
// -----------------------------------------------------------------

function stampFooters(doc: PDFKit.PDFDocument) {
  const range = doc.bufferedPageRange()
  const year = new Date().getFullYear()

  for (let i = 0; i < range.count; i++) {
    doc.switchToPage(range.start + i)

    const width = contentWidth(doc)
    const y = doc.page.height - 40
    const bottomMargin = doc.page.margins.bottom
    doc.page.margins.bottom = 0 // suppress pdfkit's auto page-break while writing in the margin band

    doc
      .moveTo(doc.page.margins.left, y - 10)
      .lineTo(doc.page.margins.left + width, y - 10)
      .lineWidth(0.75)
      .strokeColor(BORDER)
      .stroke()

    doc
      .font("Sans")
      .fontSize(8)
      .fillColor(LABEL)
      .text(
        `FirstUser by BuildNest Studio · © ${year} BuildNest Studio · Page ${i + 1}`,
        doc.page.margins.left,
        y,
        { width, align: "center", lineBreak: false }
      )

    doc.page.margins.bottom = bottomMargin
  }
}

// -----------------------------------------------------------------
// Small utilities
// -----------------------------------------------------------------

function stripProtocol(url: string): string {
  return url.replace(/^https?:\/\//, "")
}

function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", { year: "numeric", month: "long", day: "numeric" }).format(date)
}

/** Reads width/height straight from a PNG's IHDR chunk — no dependency, no decoding. */
function readPngDimensions(filePath: string): { width: number; height: number } | null {
  let fd: number | null = null
  try {
    fd = openSync(filePath, "r")
    const header = Buffer.alloc(24)
    readSync(fd, header, 0, 24, 0)
    if (header.toString("ascii", 1, 4) !== "PNG") return null
    const width = header.readUInt32BE(16)
    const height = header.readUInt32BE(20)
    if (!width || !height) return null
    return { width, height }
  } catch {
    return null
  } finally {
    if (fd !== null) closeSync(fd)
  }
}
