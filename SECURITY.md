# Security Policy

## Supported versions

We fix security problems in the latest release, on the `main` branch, and in the live website
<https://hydrologymip.org>.

## How to report a vulnerability

Please **do not** open a public issue for a security problem.

1. Use GitHub's private reporting: go to the repository's **Security** tab and choose
   **Report a vulnerability**. Only the maintainers can see the report.
2. If you cannot use GitHub, contact the project lead, [@grey-nearing](https://github.com/grey-nearing),
   through GitHub and ask for a private channel.

Please include:

- what is affected (file, endpoint, or workflow);
- the steps to reproduce the problem;
- what an attacker could do;
- any fix you suggest.

## What is in scope

- The web portal and its API, especially endpoints that accept uploads or write data (for example
  `/api/score` and `/api/submit`).
- Leaderboard integrity: any way to add, change, or fake a leaderboard entry or score without
  maintainer review.
- The GitHub Actions workflows and the dependency and supply chain of this repository.
- Credentials, tokens, or private paths exposed in the repository or in published data.

## What is not in scope

Scientific disagreements about models, metrics, or rankings are welcome as normal GitHub issues.

## What happens next

- We aim to acknowledge your report within 5 working days.
- We will keep you informed while we fix the problem.
- We will credit you in the release notes, unless you prefer not to be named.

We will not take action against anyone who reports a problem in good faith and follows this policy.
