# TODOs

This directory contains one markdown file per TODO/feature proposal.

## Active TODOs

1. [parallel-gpu-search.md](./parallel-gpu-search.md) - Parallelize world model search across multiple GPUs for ~K× speedup

## Format

Each TODO file should include:

- **Problem statement** - What limitation/issue does this address?
- **Current state** - How does the code work now?
- **Proposal** - Detailed design of the solution
- **Implementation notes** - Files to modify, API changes, etc.
- **Challenges** - Known difficulties and how to address them
- **Priority** - High/Medium/Low
- **References** - Links to relevant code (file_path:line_number)

## Lifecycle

When a TODO is completed:
- Move it to `docs/implemented/` with implementation notes
- Add a summary to the main project changelog/release notes
