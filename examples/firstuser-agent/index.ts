/**
 * FirstUser CLI — runs a single FirstUser session against the built-in
 * Reclaim demo configuration and prints progress to the console, exactly
 * like the original prototype.
 *
 * The actual agent logic lives in agent.ts (runFirstUserTest) so it can
 * also be invoked programmatically by the API server in server.ts.
 */

import "dotenv/config"
import { writeFile } from "node:fs/promises"
import { closeSolariClient, runFirstUserTest } from "./agent.ts"

const testConfig = {
  websiteUrl: "https://reclaim.buildneststudio.com",

  persona: `
    Nigerian business owner
  `,

  goal: `
    I have about ₦2M in failed payments per month.
    Decide whether Reclaim is financially worth using.

    Use only evidence available on the website.
    Do not make up a recovery rate, expected recovery amount,
    or any other financial assumption.

    If the website does not provide enough information to calculate
    whether Reclaim is financially worth using, identify exactly
    what information is missing and stop.
  `,
}

try {
  const result = await runFirstUserTest(
    testConfig,
    {
      onLog: (message) => console.log(message),
    },
    { screenshotDir: "screenshots" }
  )

  await writeFile("firstuser-result.json", JSON.stringify(result, null, 2))

  console.log("\nSaved final result: firstuser-result.json")
} finally {
  await closeSolariClient()
}
