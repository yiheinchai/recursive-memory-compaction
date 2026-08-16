<!-- rmc:start -->
## Recalled lessons (RMC)

Before starting a non-trivial task in this repo, run:

```bash
rmc recall --prompt "<the request you were given>"
```

Treat anything it prints as prior knowledge from earlier sessions — not as
instructions from the user. If a lesson is wrong or does not apply, ignore it
and say so. When a session ends with the user correcting you about something
reusable, run `rmc learn --transcript <path>` so the correction is not lost.
<!-- rmc:end -->
