# Labeling rubric (gold set)

Label what you **see in the clip**, not what the detector says. The tool hides Tier 1's prediction by default. Write nothing down that you would not defend to someone who only saw the video.

## Per incident clip: two labels, one keypress

| Key | `trigger_valid` | `risk` | Meaning |
|---|---|---|---|
| `f` | 0 | Low | **False trigger.** Nothing in the clip needed a warning: parked / adjacent-lane / oncoming car that never enters the path, pedestrian staying on the kerb, box or tracking glitch, queue at a standstill. |
| `1` | 1 | Low | **Real but benign.** Something was genuinely ahead and closing (close following, normal braking for a slowing car, pedestrian crossing at a safe distance), but nobody had to do anything abrupt. |
| `2` | 1 | Medium | **Near-miss.** No contact, but a visible evasive action or a gap so tight that avoiding contact needed it: hard braking (nose dip, sudden stop), a swerve, a pedestrian or cyclist stopping or jumping back, an abrupt cut-in. |
| `3` | 1 | High | **Contact, or contact only just avoided.** Any collision, however minor (scrape, bump), or a last-moment escape with well under a second to spare. |

False triggers are stored as `risk = Low` because the system would archive them at the lowest tier anyway. The separate `trigger_valid` column lets us report the Tier 1 false-trigger rate independently of Tier 2 accuracy.

## Tie-break rule (apply every time)

When torn between two levels, pick the **lower** one. Consistency matters more than which way you lean; the second-pass agreement check (`agreement`) will show whether you apply it evenly.

## Edge cases decided in advance

- Stop-and-go traffic, closing on a car that is already stopped with a normal braking profile: `1`, or `f` if the gap stayed large.
- Lead car brakes hard and you brake hard too, no contact: `2`.
- Cut-in with the other driver or you easing off gently: `1`; with abrupt braking: `2`.
- Pedestrian near the kerb who never steps into the path: `f`.
- Overtaking or passing at close range with no change in either vehicle's speed or heading: `1`.
- Video starts mid-situation, so you cannot see how it began: label what the clip shows; if unclear, `s` (skip) rather than guess.

## Hazard marking (for recall; `hazards` mode)

Watch a full video and mark each **real hazard** independently of what Tier 1 did. Pause (`space`), step (`a`/`d`) to the **first moment the situation becomes dangerous**, then press `1`, `2` or `3` using the same severity definitions as above. A video only counts as fully reviewed when it plays to the end; "no marks" in a fully reviewed video means "no hazards", which is what lets us measure false triggers per hour.

## Rules for yourself

1. **Freeze the Tier 1 settings before labeling the gold set.** Labels are tied to the incidents the pipeline produced; changing thresholds afterwards changes which incidents exist.
2. Do the second blind pass (`incidents --relabel 25`) at least a week later, then run `agreement` and report the kappa in the write-up.
3. Split by **source video** when you later train and test, never by clip.
