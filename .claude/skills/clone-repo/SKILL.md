---
name: clone-repo
description: Clones the bhoga01ai/langchain-agents repo (https://github.com/bhoga01ai/langchain-agents.git) to a local destination. Use this whenever the user asks to clone "my repo," "the langchain-agents repo," or otherwise wants a fresh local copy of this specific project, even if they don't give the full URL themselves.
---

# Clone Repo

Clones this repo:

```
https://github.com/bhoga01ai/langchain-agents.git
```

## Steps

1. If the user hasn't given a destination path, ask where to clone it (default to the current directory if they have no preference).
2. Check the destination doesn't already contain a non-empty directory that would conflict — if it does, confirm with the user before overwriting anything.
3. Run:
   ```bash
   git clone https://github.com/bhoga01ai/langchain-agents.git <destination>
   ```
4. Report the resulting local path back to the user. Do not run any install/setup steps unless the user asks for them — this skill only clones.
