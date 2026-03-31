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

## Follow-up UX risks / open questions
- Run detail still includes advanced geometry/trend sections that may be too dense for first-time users.
- Validation and onboarding remain in the same shell; may need role-based access splitting.
- Need usage analytics to confirm if users complete top-3 jobs faster after this pass.
