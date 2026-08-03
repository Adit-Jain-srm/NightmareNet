---
name: grill
description: >-
  Use when about to create a skill, start a feature, design a system, or plan
  any non-trivial work. Also use when user says grill me, question my plan,
  challenge this, stress-test, or validate my thinking. Prevents building the
  wrong thing by resolving decisions before implementation.
---

## Overview

Resolve every decision branch before committing to implementation. Ask questions ONE AT A TIME until the full picture emerges.

## How

Ask me questions ONE AT A TIME about what I'm about to build. For each question, provide your recommended answer. If a question can be answered by exploring what already exists, explore first.

Cover these branches:

```
1. WHO is this for? (specific user, not "developers")
2. WHAT specific pain does this eliminate? (named, felt, recurring)
3. WHY doesn't a solution already exist? (searched? what did you find?)
4. WHAT'S the simplest version that would be useful? (not the dream — the MVP)
5. HOW would someone find this? (discovery path — search terms, marketplace, word of mouth?)
6. WHAT makes someone SWITCH from their current approach? (the switching cost must be worth it)
7. WHAT could go wrong? (failure modes, edge cases, misuse)
8. HOW do you KNOW it works? (verification — not "I think" but "I proved")
```

## Persistence

ACTIVE until all branches resolved. Don't let me start building until questions are answered. If I try to skip ahead: "We haven't resolved [branch]. Answer first."

## Depth Control

- Simple task (< 5 min): 3-4 questions
- Medium task (< 1 hour): 5-8 questions  
- Complex task (> 1 hour): full tree, multiple sub-branches per question

## When to Stop

Stop grilling when:
- Every branch has a clear answer (not "we'll figure it out later")
- I can state in ONE sentence what I'm building, for whom, and why it's better
- There's a concrete first step (not vague "start coding")

## Anti-Pattern

"Let me just start and see where it goes" ← This is the PROBLEM grilling prevents. Every hour spent grilling saves 10 hours of building the wrong thing.

## Example Exchange

```
Agent: "WHO is this for? My recommendation: solo developers working on 
       side projects who hit the same deployment friction every weekend."
User:  "Actually it's for teams — 3-5 devs sharing staging environments."
Agent: "Good — that changes scope significantly. WHAT specific pain does 
       this eliminate? My recommendation: 'We broke staging again because 
       two people deployed at the same time.'"
```

Key: ONE question, recommend an answer, let the user correct. Don't ask three things at once.

> "No-one knows exactly what they want." — The Pragmatic Programmer
>
> The fix is to DISCOVER what you want through structured interrogation before committing to implementation.
