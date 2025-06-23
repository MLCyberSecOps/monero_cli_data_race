# ThreadGuard Research & Development Archive

This directory contains historical versions and research artifacts of the ThreadGuard static analyzer, documenting its evolution and experimental features.

## 📂 Version History

### v2.0.0 (Current)
- **Enhanced Analysis Engine**
  - Improved pattern matching for C++ concurrency primitives
  - Better handling of complex locking scenarios
  - Reduced false positive rate

### v1.1.0
- **Initial Release**
  - Basic data race detection
  - Simple lock analysis
  - Monero-specific patterns

## 🔍 Directory Structure

```
research_exhaust/
├── threadguard/
│   └── v1/                     # Initial research versions
│       ├── threadguard_new.py   # Early development version
│       └── threadguard_fixes.py # Experimental fixes and enhancements
└── README.md                    # This file
```

## 🛠️ Development Workflow

1. **Experimental Features**
   - New features are developed in the research directory
   - Tested against known concurrency patterns
   - Performance benchmarks are recorded

2. **Promotion to Main**
   - Well-tested features move to the main codebase
   - Documentation is updated
   - Version number is incremented

## 📊 Research Metrics

| Version | Features | Test Coverage | Performance | Status       |
|---------|----------|---------------|--------------|--------------|
| 2.0.0   | Advanced | 85%           | Fast         | Production   |
| 1.1.0   | Basic    | 65%           | Moderate     | Deprecated   |


## 🎯 Purpose

This directory serves as a reference for:
- Historical tracking of development
- Research experiments and prototypes
- Performance comparisons between versions
- Reference implementations of complex algorithms
- Educational examples of concurrency patterns

## ⚠️ Usage Notice

Files in this directory are for research and development purposes only. They may:
- Contain experimental code
- Have known issues
- Lack proper documentation
- Not be production-ready

**For production use**, always use the latest stable version from the root directory (`threadguard_enhanced.py`).

## 🤝 Contributing

We welcome contributions to the research efforts. When contributing:

1. **Create a new version**
   ```bash
   mkdir -p threadguard/vX.Y.Z  # Use semantic versioning
   ```

2. **Document your changes**
   - Update this README
   - Include a CHANGELOG.md in your version directory
   - Document any new patterns or analysis techniques

3. **Include tests**
   - Add test cases for new features
   - Include performance benchmarks
   - Document any limitations or known issues

4. **Submit a Pull Request**
   - Reference related issues
   - Include before/after performance metrics
   - Update documentation as needed

## 📝 Versioning Policy

- **Major versions (X.0.0)**: Major architectural changes
- **Minor versions (1.X.0)**: New features, backwards compatible
- **Patch versions (1.0.X)**: Bug fixes and minor improvements
