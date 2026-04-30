# Project state and learnings

## Git workflow

- After making changes, always ask if you should commit to git. If the user says yes, always run `git add` and `git commit` with a clear, descriptive commit message.
- Commit after each meaningful unit of work (e.g., completing a service, adding a handler, finishing a hook). Do NOT batch all changes into one giant commit.
- Never paste API keys / secrets in chat. The user has had to rotate their OpenAI key twice — paste keys directly on the server terminal, never here.