---
name: "🐛 Monero Code Issue"
about: "Report a potential issue found in Monero's codebase using this tool"
title: "[Monero] "
labels: "monero-issue, needs-triage"
---

⚠️ **Before You Begin**
This issue tracker is for issues with the ThreadGuard analysis tool itself. 

If you've found a potential issue in Monero's code:
1. First report it at: [Monero GitHub Issues](https://github.com/monero-project/monero/issues/new/choose)
2. Then, you may reference it here for tracking purposes.

---

## Monero Issue Reference

- **Monero Issue Link**: [Link to Monero issue or PR]
- **Monero Commit Hash**: [e.g., `a1b2c3d`]
- **ThreadGuard Version**: [e.g., 2.0.0]

## Analysis Details

### Issue Description
[Briefly describe the concurrency issue found]

### Affected Files
- `path/to/file1`
- `path/to/file2`

### Steps to Reproduce
1. How to run the analysis:
   ```bash
   python threadguard_enhanced.py --file src/monero/file.cpp
   ```
2. Specific flags or configurations used

### Tool Output
```
[Paste relevant output from the tool]
```

## Additional Context
- Any additional observations
- Potential impact
- Suggested fixes (if any)

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.0]
- Monero Branch/Commit: [e.g., master@a1b2c3d]
