/**
 * Minimal in-memory per-IP rate limiter for POST /api/tests.
 *
 * Each investigation launches a real Solari browser session and up to
 * MAX_LLM_CALLS OpenAI calls, so the publicly reachable demo needs a cheap
 * cap against unbounded usage. No database/Redis — this assumes a single
 * process instance, matching the rest of FirstUser's in-memory session
 * store (see server.ts).
 */

import type { NextFunction, Request, Response } from "express"

const WINDOW_MS = 60 * 60 * 1000
const MAX_REQUESTS_PER_WINDOW = 3

const hitsByIp = new Map<string, number[]>()

function getClientIp(req: Request): string {
  return req.ip ?? req.socket.remoteAddress ?? "unknown"
}

export function rateLimitTestCreation(req: Request, res: Response, next: NextFunction) {
  const ip = getClientIp(req)
  const now = Date.now()
  const recent = (hitsByIp.get(ip) ?? []).filter((timestamp) => now - timestamp < WINDOW_MS)

  if (recent.length >= MAX_REQUESTS_PER_WINDOW) {
    res.status(429).json({
      error: "Too many investigations from this IP. Please try again in a while.",
    })
    return
  }

  recent.push(now)
  hitsByIp.set(ip, recent)
  next()
}
