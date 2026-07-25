# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Instead, please email: **aditjain2005@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if you have one)

We will acknowledge receipt within 48 hours and provide a timeline for a fix within 7 days. Critical vulnerabilities affecting production deployments will be patched within 72 hours.

## Scope

The following are in scope for security reports:
- Authentication bypass in the API layer
- Secrets exposure (API keys, tokens)
- Remote code execution via distortion inputs
- Dependency vulnerabilities with known exploits
- CORS misconfiguration allowing credential theft

The following are out of scope:
- Denial of service via resource exhaustion (rate limiting exists)
- Attacks requiring physical access to the machine
- Social engineering attacks
