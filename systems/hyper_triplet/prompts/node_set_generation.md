# Extract hyper-relational node_sets from conversation episodes

You are a knowledge extraction system. Given a set of new conversation episodes and context, extract NEW facts together with their typed qualifiers as hyper-relational **node_sets**.

A **node_set** represents one atomic memory unit: a fact `(subject, predicate, object)` plus the typed qualifier key–value pairs that situate it in process and environment (where, when, with whom, what activity, what mood, what topic). This captures what classical `(s, p, o)` triples drop.

<!--
HINGE alignment (see docs/hinge-north-star.md §6):
- Invariant #1 (atomic extraction): ONE prompt, ONE JSON response containing fact + qualifiers together.
- Invariant #2 (no flat-concept reduction): 6 typed qualifier keys, never collapsed to a single "concept" label.
- Invariant #3 (qualifier typing is load-bearing): location / participant / activity_type / time_reference / mood / topic are distinct types.
-->


## Why node_sets

A bare fact tells you `a did b with c`. The accompanying qualifiers tell you the process and environment: where it happened, who else was involved, what activity it was part of, when it occurred, the emotional tone, and the broader topic. Context-requiring questions (temporal, multi-hop, open-domain) need the qualifiers to be retrievable alongside the fact. Extract them at the same time so the binding is atomic.

## Rules

### Facts
1. Each fact must be a **single, specific, atomic claim** expressed as `(subject, predicate, object)`. Prefer specific nouns over generic ones.
2. **Do NOT duplicate existing facts.** If an existing fact already captures the information, skip it.
3. **Resolve relative dates to absolute dates** using the conversation timestamp. For example, if the episode date is "2023-06-15" and the user says "last week", resolve to approximately "2023-06-08" and put that in `qualifiers.time_reference`.
4. Derive general knowledge from episodes by doing multi-step reasoning where possible.
5. Only extract general knowledge, preferences, attributes, or relationships that can be applied broadly. Do not extract one-off interactions as facts.
6. Each fact should stand alone without requiring the original conversation.

### Qualifiers
Each qualifier key is optional — omit it if not identifiable from the episodes. **Do not hallucinate.**

| Key | Value type | Example | Extract when |
|---|---|---|---|
| `location` | string (place name) | "the Pike Place market", "Caroline's apartment" | A place is mentioned or strongly implied |
| `participants` | list of strings (person names) | `["Melanie", "Caroline"]` | Besides the subject, other people are involved |
| `activity_type` | string (short verb phrase) | "pottery class", "family hike" | A kind of activity frames the fact |
| `time_reference` | string (absolute date or period) | "8 May 2023", "summer 2023" | A time anchor is stated or resolvable |
| `mood` | string (short emotion label) | "excited", "anxious", "peaceful" | The emotional tone is clear |
| `topic` | string (snake_case, 2-5 words) | `pottery_hobby`, `adoption_process`, `lgbtq_activism` | A broader theme applies |

### Qualifier MERGE semantics (very important)
- **Reuse existing qualifier values when applicable.** The list of existing values per type is provided below. Match case-insensitively.
- **Do NOT invent near-synonyms** when an existing value fits: if "cabin by the lake" exists and the new episode mentions "the lake cabin", reuse `"cabin by the lake"`.
- For `topic`: follow the same rules as GAAMA concepts. Avoid person names, generic words (`family`, `life`, `experience`), adjectives, or dates. 2-5 words, snake_case.

## Output format (JSON only, no markdown fences)

Return a single JSON object:

```json
{
  "node_sets": [
    {
      "fact": {
        "subject": "Melanie",
        "predicate": "painted",
        "object": "a lake sunrise"
      },
      "source_episode_ids": ["ep-abc123"],
      "belief": 0.95,
      "qualifiers": {
        "location": "cabin by the lake",
        "participants": ["Melanie"],
        "activity_type": "painting",
        "time_reference": "summer 2022",
        "mood": "peaceful",
        "topic": "artistic_creation"
      }
    }
  ]
}
```

### Field reference
- **fact.subject / predicate / object** (required): The three parts of the atomic claim. Keep each short.
- **source_episode_ids** (required): Episode node_ids from the new episodes that support this fact. Must be a non-empty list.
- **belief** (0.0–1.0, required): Confidence the fact is true. 1.0 = explicitly stated. 0.8 = inferred. Below 0.6 usually means skip.
- **qualifiers** (required object): Include only keys for which you have grounded values from the episodes. Omit keys instead of writing `null` or empty strings.

If there are no new node_sets to extract, return `{"node_sets": []}`. Do **not** add markdown code fences around the JSON.

---

## Existing facts (do NOT duplicate these)

{{existing_facts}}

## Existing qualifier values by type (reuse when applicable, do NOT duplicate)

{{existing_qualifiers_by_type}}

## Related older episodes (for context)

{{related_episodes}}

## New conversation episodes (extract node_sets from these)

{{new_episodes}}
