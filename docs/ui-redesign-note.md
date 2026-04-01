# UI Redesign Note (Brutal Compression)

## One-sentence product definition
This app helps operations teams detect system risk, upload fresh telemetry, and investigate run outcomes in minutes so they can decide the next action without guesswork.

## Top 3 user jobs (product compression)
1. **Monitor risk now** (current state + risk + recommendation).
2. **Upload telemetry fast** (clear required inputs and immediate progress/error feedback).
3. **Investigate a run** (find run, inspect timeline, confirm why risk changed).

## Keep / Merge / Kill / Rebuild

### Keep
- Core pages: Dashboard, Upload, Runs, Run Detail.
- Existing API integrations and business logic.
- Existing risk/geometry computations.

### Merge
- Treat Dashboard + run CTA as one operational starting point.
- Treat run list as direct entry to run investigation.
- Keep Validation and Onboarding as secondary setup flows, not primary nav focus.

### Kill
- Marketing-like copy and non-obvious labels.
- Redundant action groups competing for attention.
- Ambiguous helper text that doesn’t explain next action.

### Rebuild
- Navigation labels rewritten as explicit user jobs.
- Page titles/subtitles rewritten for plain-language intent and expected next step.
- Visual system overridden with a minimal operational style (flat panels, strict spacing, clear controls).
- Empty/failure guidance rewritten to tell users what to do next.

## New screen map (replacement IA)

### Primary nav
1. **Monitor risk** (`/dashboard`) — primary CTA: **Upload telemetry now**.
2. **Upload telemetry** (`/upload`) — primary CTA: **Start upload**.
3. **Investigate run** (`/app/runs`) — primary CTA: **Open a run**.

### Secondary nav
- **Onboarding setup** (`/onboarding`) — setup only.
- **Validation replay** (`/validation`) — non-primary workflow.

## UI system standardization
- **Typography:** fewer size jumps, strong page title + compact subtitles.
- **Spacing:** tokenized 4/8/12/16/24/32 rhythm.
- **Buttons:** one dominant primary action style, secondary outlined style.
- **Cards/Panels:** flat white panel + 1px border.
- **Forms/Tables:** compact controls, explicit labels, no decorative chrome.
- **Alerts/Errors:** bordered message blocks with actionable failure language.
- **Empty states:** each includes “what to do next”.
- **Loading/Progress:** concise status messaging oriented around action completion.

## Second-pass structural hardening (implemented)
- Consolidated fragmented stylesheet overrides into a single `styles.css` so layout/visual rules live in one place.
- Simplified Investigate run IA: run creation moved from top-level panel into inline fallback inside run list.
- Rewrote remaining setup labels from legacy product language to explicit operator-job language.
- Removed unused static asset remnants and dead boot wiring linked to non-existent controls.

## Remaining UX risks / open questions
- Run detail still contains dense geometry detail and can overwhelm first-time operators.
- Client logic is spread across large JS files; a failure inside one large module can still disrupt multiple sections.
