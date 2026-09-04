# FirstUser

**An autonomous AI agent that visits a website like a real first-time user and determines whether the available evidence is sufficient to answer a real-world question.**

🌐 **Live demo:** https://firstuser.buildneststudio.com

> FirstUser runs real browser investigations using Solari and OpenAI. LIVE investigations explore the website in a real cloud browser and generate an evidence-based report.

FirstUser does not simply crawl pages or check whether UI elements work.

It receives:

- a website URL
- a user persona
- a mission/question

Then it uses an LLM-guided exploration loop and a real Solari Cloud Browser session to:

1. Observe the website.
2. Decide what information matters.
3. Navigate, click, expand UI elements, and scroll through below-the-fold content.
4. Extract evidence from what it actually encounters.
5. Continue exploring within a strict LLM budget.
6. Reserve one final reasoning call for evaluation.
7. Produce an evidence-based verdict.

## Real browser exploration

FirstUser uses [Solari](https://getsolari.com) Cloud Browser to interact with the actual website rather than analyzing a placeholder or static HTML snapshot. Every click, scroll, and expandable toggle happens in a live, real browser session against the real site.

## Observe → Decide → Act → Observe

The agent runs a tight loop instead of summarizing a page once:

1. **Observe** the current viewport — visible headings, text, buttons, links, and expandable elements, plus scroll position and page height.
2. **Decide** the single most useful next action given the visitor's persona and goal.
3. **Act** through a constrained, safe action space (`click_button`, `click_link`, `toggle_expandable`, `scroll`, or `none`).
4. **Observe again** — re-read the page, extract any new evidence, and repeat.

This is why FirstUser can find things a one-shot page summary misses: it behaves like a person actually looking around.

## Evidence-based decisions

The agent must not invent information. Every claim in the final verdict must be traceable to something it actually observed during the session — no outside knowledge, no assumed numbers, no filled-in gaps.

If the website does not provide enough evidence to answer the visitor's goal, the correct answer is:

```text
UNCERTAIN
```

Missing evidence is treated as a first-class finding, not a failure of the agent.

## Navigation is not the entire information architecture

A website's navigation menu is not a complete map of its information. Meaningful evidence often lives below the fold, in landing-page sections, cards, feature grids, testimonials, near calls-to-action, or in the footer — not just behind a nav link.

FirstUser's reasoning treats the current page as a source of evidence in its own right: it can scroll through below-the-fold content when that's likely to reveal something relevant, rather than only ever following navigation links.

## Safe exploration

The agent never:

- fills out or submits forms
- completes purchases
- logs in, signs up, or creates an account
- sends messages
- modifies user data

If a form or input field appears, FirstUser does not interact with it. This rule is absolute and does not change based on the mission.

## LLM budget

Exploration runs under a bounded LLM budget (`MAX_LLM_CALLS`, currently 12 calls per session). One call is always reserved for the final evaluation, so exploration itself is capped at `MAX_LLM_CALLS - 1` calls. If the budget is exhausted mid-exploration, the agent stops cleanly and still produces a final evaluation from whatever evidence it collected.

## Architecture

```text
User Mission
     ↓
FirstUser Agent
     ↓
Observe → Decide → Act → Observe
     ↓
Solari Cloud Browser
     ↓
Real Website
     ↓
Evidence Collection
     ↓
Final Evaluation
     ↓
Evidence-based Verdict
```

The final evaluator receives the structured observation history, action history, and extracted evidence. It is instructed not to use outside knowledge or invent missing numbers, and can return `yes`, `no`, or `uncertain`.

## Supported actions

The action space is intentionally small:

- `click_button` clicks a visible observed button.
- `click_link` clicks a visible observed link using the observed DOM mapping.
- `toggle_expandable` opens or closes visible expandable content such as FAQ items.
- `scroll` scrolls up or down by a bounded amount to reveal more of the current page.
- `none` stops exploration when no useful action remains.

Expandable content detection supports:

- native HTML `summary` elements inside `details`
- accessible accordion buttons with `aria-expanded`
- accessible accordion buttons with `aria-controls`

## Example journey

The included sample run evaluates `https://reclaim.buildneststudio.com` for this goal:

> I have about ₦2M in failed payments per month. Decide whether Reclaim is financially worth using. Use only evidence available on the website.

In the sample journey, FirstUser:

- visited the pricing section
- found plan fees and recovery-share percentages
- navigated to the FAQ
- opened the FAQ item "What happens if you don't recover anything?"
- extracted the revealed answer about Pay-on-Recovery and paid-tier downside risk
- visited "How it works"
- concluded `uncertain` because the site did not provide a quantified recovery rate or expected recovered amount

See:

- [sample output](examples/sample-output/firstuser-result.json)
- [example screenshots](screenshots/example-run)

## How to run

Requirements:

- Node.js 20 or newer
- a Solari API key
- an OpenAI API key

From the repository root:

```bash
cd examples/firstuser-agent
npm install
```

Create an environment file:

```bash
cp .env.example .env
```

Then fill in:

```bash
SOLARI_API_KEY=your_solari_api_key
OPENAI_API_KEY=your_openai_api_key
```

Run the agent as a one-off CLI script (uses the built-in Reclaim demo config):

```bash
npm start
```

The run writes:

- `firstuser-result.json`
- screenshots in `screenshots/`

Those runtime outputs are gitignored by default. Curated examples live under `examples/sample-output/` and `screenshots/example-run/`.

### Running the web UI

The same agent can be driven from a browser instead, entering any website/persona/goal:

```bash
npm run server
```

Then open `http://localhost:3000`. The server starts a real FirstUser session per request, exposes safe progress state for the Live Exploration screen, and serves the final evaluation to the Report screen. It never sends `SOLARI_API_KEY` or `OPENAI_API_KEY` to the browser.

### Try the deployed version

Visit:

https://firstuser.buildneststudio.com

Architecture:

- [agent.ts](agent.ts) — the core agent (`runFirstUserTest(config, hooks)`), extracted so it can be invoked programmatically. Exploration logic, evidence rules, the final evaluator, and the LLM budget live here.
- [index.ts](index.ts) — thin CLI wrapper around `agent.ts` for `npm start`.
- [server.ts](server.ts) — Express API (`POST /api/tests`, `GET /api/tests/:id/status`, `GET /api/tests/:id/result`, `GET /api/tests/:id/screenshots/:file`) plus static hosting of `frontend/`. Sessions are tracked in memory (no database — this is a V1).
- [frontend/FirstUser.dc.html](frontend/FirstUser.dc.html) — the UI. "Start investigation" always launches a real backend session (badged `LIVE SESSION`); "Watch a recorded demo instead" runs the original scripted walkthrough (badged `DEMO SESSION`). The two never share data.

## Configuration

For the CLI, the test configuration (`websiteUrl`, `persona`, `goal`) is defined in [index.ts](index.ts). For the web UI, it comes from the Create screen's website/persona/mission fields at request time instead.

`MAX_STEPS` and `MAX_LLM_CALLS` (the exploration step limit and LLM budget) live in [agent.ts](agent.ts) and apply to both.

## Limitations

FirstUser is a prototype, not a complete browser automation framework.

Current limitations:

- It only sees and reasons from the current viewport.
- It uses a small fixed action set.
- It does not handle authentication flows.
- It does not fill or submit forms.
- It does not handle modals or arbitrary custom widgets.
- It depends on the model returning valid JSON.
- It can miss evidence that is hidden behind unsupported interaction patterns.
- It should not be used as the only source of truth for high-stakes decisions.

## Built with

- [Solari](https://getsolari.com) for the cloud browser session
- [OpenAI](https://openai.com) for reasoning and evidence extraction
- TypeScript
- `tsx`
