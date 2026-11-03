# TEST_PLAN addendum B — identity ceiling audit (frozen 2026-11-02, before the audit is scored)

## Why
On the 9B test set Rater A gave I = 4 to 58 / 60 images. QWK / rho cannot rank a constant column (kappa paradox,
Feinstein & Cicchetti 1990), so agreement is reported as exact agreement and Gwet's AC1 instead
(I: gpt-4o 88 % / 0.88, 32B v3 57 % / 0.52, gpt-6.1-sol 43 % / 0.37). Whether the "all 4" ceiling is real, or a
lenient human score that gpt-6.1-sol correctly questions, is checked by a closer, attribute-level look.

## Sample (fixed, `audit_sample_20.json`, random.Random(2028))
* Set S (agreement): 10 of the 9B test images with I = 4 from Rater A, gpt-4o and gpt-6.1-sol.
* Set D (disagreement): 10 of the 9B test images with I = 4 from Rater A and I = 3 from gpt-6.1-sol.
* The 20 rows are shuffled, labelled A01–A20, and the set is not shown.

## Procedure (Rater A)
1. Attribute checklist against the reference head, each yes / no / n.a.: hair colour, hairstyle, eyes (shape and
   colour), face, signature features (accessories, markings), species / body type. Attributes the prompt asks to
   change are n.a. (prompt wins, as in the original rubric).
2. Re-scored identity 0–4 after the checklist.
3. Only then open the sheet "评委理由" (gpt-6.1-sol identity notes for all 20 rows) and mark for each row whether the
   problem it names is real (yes / partly / no / no problem named).

## Analysis
* Share of Set S still at 4 after the audit; share of Set D re-scored to ≤ 3; share of gpt-6.1-sol identity
  notes judged real, per set.
* Reading: Set S stays 4 and Set D notes mostly "no" → the ceiling is real and gpt-6.1-sol is over-strict;
  Set D notes mostly "yes" / re-scored ≤ 3 → the original human score was lenient and gpt-6.1-sol's strictness is
  justified. Both outcomes are reported. n = 10 per set; proportions with exact (Clopper–Pearson) 95 % CIs.
