# Contributing

Contributions that improve the accuracy, portability, or clarity of the HPC guidance are welcome.

## Before making a change

Search the open issues and pull requests to avoid duplicating work. For a larger change, open or reference an issue describing the problem first.

Keep public documentation free of credentials, access tokens, private data, and other sensitive information. Use placeholders in example commands where site-specific values are required.

## Where to put documentation

- Put system-specific guidance in the directory for that resource, such as `DAWN/`, `DGX-Spark/`, or `Isambard-AI/`.
- Put workflows that apply across systems in `example_workflows/`.
- Prefer extending an existing guide when it already covers the same task.

## Writing guidance

Use Markdown that renders correctly on GitHub. Prefer relative links to files in this repository and fenced code blocks for commands and configuration.

For commands that depend on a particular system, state where they were tested and call out assumptions such as scheduler, accelerator type, module environment, or container runtime when they affect reproducibility.

## Submitting a pull request

Keep each pull request focused on one change. In the description, explain what was changed, which system or workflow it applies to, and how you checked the instructions.

Before submitting, review the rendered Markdown and run:

```bash
git diff --check
```

The Research Computing Platforms team reviews contributions before they are merged.
