# Multi-Agent Code Translation System

A sophisticated multi-agent system for converting legacy scientific codebases (Fortran, C/C++, Python2) to modern Python 3, featuring intelligent documentation processing, comprehensive testing, and parallel computing constructs translation.

## Features

### 🤖 5-Agent Architecture
- **Orchestrator Agent**: Coordinates workflow, manages dependencies, handles rollbacks
- **Translation Agent**: Core translation engine with pattern recognition
- **Knowledge Agent**: RAG-powered documentation processor and context provider
- **Testing Agent**: Comprehensive validation and quality assurance
- **Web Research Agent**: External knowledge gathering and library integration

### 🔄 Language Support
- **Fortran** → Python 3 (COMMON blocks, DO loops, subroutines)
- **C/C++** → Python 3 (pointers, memory management, structs)
- **Python 2** → Python 3 (print statements, unicode, iterators)

### 🧠 Intelligent Features
- **RAG System**: Vector-based documentation search and context retrieval
- **Dependency Analysis**: Automatic dependency graph generation and resolution
- **Numerical Validation**: Accuracy testing with configurable tolerances
- **Performance Benchmarking**: Before/after performance comparison
- **Git Integration**: Automated version control with rollback capability

## Installation

### Prerequisites
- Python 3.8+
- Git
- 8GB+ RAM (for large codebases)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd translation_agent/code_translator

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for specific file types
pip install PyPDF2 python-docx beautifulsoup4  # For document processing
```

### Configuration
Copy and customize the configuration file:
```bash
cp config/default_config.yaml config/my_project_config.yaml
# Edit config/my_project_config.yaml as needed
```

## Quick Start

### Basic Translation
```bash
python main.py --project ./legacy_fortran_code --target python3
```

### With Documentation
```bash
python main.py --project ./legacy_code --docs ./documentation --target python3
```

### Advanced Usage
```bash
python main.py \
  --project ./complex_project \
  --docs ./manuals \
  --target python3 \
  --validation comprehensive \
  --jobs 8 \
  --backup \
  --config ./config/my_config.yaml
```

## Usage Examples

### Example 1: Fortran Scientific Code
```bash
# Translate Fortran numerical simulation
python main.py \
  --project ./fortran_simulation \
  --docs ./scientific_papers \
  --validation comprehensive \
  --jobs 4
```

### Example 2: C++ Library
```bash
# Translate C++ math library with comprehensive testing
python main.py \
  --project ./cpp_math_lib \
  --docs ./api_documentation \
  --validation comprehensive \
  --backup
```

### Example 3: Python 2 Migration
```bash
# Migrate Python 2 project
python main.py \
  --project ./old_python_project \
  --target python3 \
  --validation standard
```

## Configuration

### Key Configuration Options

#### Translation Settings
```yaml
translation:
  strategy: "conservative"  # conservative, aggressive, hybrid
  preserve_comments: true
  generate_type_hints: true
  modernize_syntax: true
  
  numerical_tolerance:
    rtol: 1e-12
    atol: 1e-15
```

#### Agent Settings
```yaml
agents:
  orchestrator:
    max_retries: 3
    checkpoint_interval: 100
    
  knowledge:
    vector_store: "faiss"
    embedding_model: "all-MiniLM-L6-v2"
    chunk_size: 1000
    
  testing:
    test_coverage_threshold: 0.8
    numerical_validation: true
```

#### Performance Settings
```yaml
performance:
  memory_limit_mb: 4096
  cpu_cores: -1  # Use all available
  parallel_processing: true
```

## API Usage

### Programmatic Interface
```python
from code_translator.main import MultiAgentTranslator

# Initialize translator
translator = MultiAgentTranslator(config_path="config/my_config.yaml")

# Load project and documentation
translator.load_project("/path/to/legacy/code")
translator.load_documentation("/path/to/docs/")

# Start translation
result = translator.translate(
    target_language="python3",
    parallel_jobs=4,
    validation_level="comprehensive",
    backup_original=True
)

# Generate report
report = translator.generate_report()
```

### Custom Agent Integration
```python
from code_translator.agents.orchestrator import OrchestratorAgent
from code_translator.utils.config_manager import ConfigManager

config = ConfigManager("config/my_config.yaml")
orchestrator = OrchestratorAgent(config)

# Register custom agent
orchestrator.register_agent("custom_agent", my_custom_agent)
```

## Architecture

### Agent Communication Flow
```
Orchestrator Agent
    ├── Analyzes project structure
    ├── Creates dependency graph
    ├── Coordinates translation workflow
    └── Manages state and rollbacks

Translation Agent
    ├── Detects source language
    ├── Parses code structures
    ├── Applies translation patterns
    └── Generates Python code

Knowledge Agent
    ├── Processes documentation (PDF, DOCX, MD)
    ├── Builds vector embeddings
    ├── Provides contextual guidance
    └── Suggests best practices

Testing Agent
    ├── Validates syntax and imports
    ├── Runs numerical accuracy tests
    ├── Benchmarks performance
    └── Generates test reports
```

### Directory Structure
```
code_translator/
├── agents/              # Multi-agent system components
├── parsers/             # Language-specific parsers
├── translators/         # Translation engines
├── testing/             # Validation and testing framework
├── knowledge/           # RAG system and document processing
├── utils/               # Utilities and configuration
├── templates/           # Translation patterns and templates
├── config/              # Configuration files
└── main.py             # Main entry point
```

## Translation Patterns

### Fortran → Python
```fortran
! Fortran DO loop
DO I = 1, N
    A(I) = B(I) + C(I)
END DO
```
```python
# Translated Python
for i in range(1, n + 1):
    a[i-1] = b[i-1] + c[i-1]  # Converted to 0-based indexing
```

### C++ → Python
```cpp
// C++ memory allocation
double* array = (double*)malloc(n * sizeof(double));
for(int i = 0; i < n; i++) {
    array[i] = sin(i * 0.1);
}
free(array);
```
```python
# Translated Python
import numpy as np
import math

array = np.zeros(n, dtype=float)  # Modern numpy approach
for i in range(n):
    array[i] = math.sin(i * 0.1)
# Python handles memory management automatically
```

## Testing and Validation

### Validation Levels

#### Basic
- Syntax validation
- Import checking
- Basic static analysis

#### Standard (Default)
- All basic checks
- Unit test generation
- Integration testing
- Performance profiling

#### Comprehensive
- All standard checks
- Numerical accuracy validation
- Cross-platform testing
- Memory usage analysis

### Numerical Accuracy Testing
The system automatically generates test cases and compares outputs:
```python
# Configurable tolerances
numerical_tolerance:
  rtol: 1e-12  # Relative tolerance
  atol: 1e-15  # Absolute tolerance
```

### Performance Benchmarking
Compares execution time, memory usage, and reliability:
```
Performance Report:
├── Execution Time: 15% faster
├── Memory Usage: 23% more efficient  
├── Success Rate: 100% → 100%
└── Numerical Accuracy: ✓ All tests passed
```

## Advanced Features

### Git Integration
```yaml
git:
  auto_commit: true
  commit_frequency: "per_file"
  branch_naming: "translation_{timestamp}"
  create_pull_request: false
```

### Parallel Processing
```yaml
system:
  max_parallel_jobs: 8
  parallel_conversion: true
```

### Custom Translation Patterns
Add custom patterns in `templates/translation_patterns/`:
```python
custom_patterns = {
    'my_fortran_pattern': {
        'pattern': r'CALL\s+MY_ROUTINE\s*\(([^)]*)\)',
        'replacement': 'my_python_function({args})',
        'flags': re.IGNORECASE
    }
}
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Reduce memory usage
export TRANSLATION_MAX_JOBS=2
python main.py --project ./large_project --jobs 2
```

#### Dependency Errors
```bash
# Install missing dependencies
pip install numpy scipy matplotlib
# Or install all optional dependencies
pip install -r requirements.txt
```

#### Performance Issues
```bash
# Enable performance optimizations
python main.py --project ./code --config config/performance_config.yaml
```

### Debug Mode
```bash
python main.py --project ./code --verbose --dry-run
```

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black code_translator/
flake8 code_translator/
```

### Adding New Language Support
1. Create parser in `parsers/new_language_parser.py`
2. Create translator in `translators/new_language_to_python.py`
3. Add patterns in `templates/translation_patterns/`
4. Update configuration options

## Performance Considerations

### Large Codebases
- Use parallel processing (`--jobs 8`)
- Enable incremental processing
- Consider memory limits for vector embeddings

### Memory Usage
- Vector store: ~100MB per 1000 documents
- Translation cache: ~50MB per 1000 functions
- Git operations: Minimal overhead

### Recommended Hardware
- **CPU**: 8+ cores for optimal parallel processing
- **Memory**: 16GB+ for large projects with documentation
- **Storage**: SSD recommended for git operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include configuration files and error logs

## Citation

If you use this system in academic work, please cite:
```bibtex
@software{multiagent_code_translator,
  title={Multi-Agent Code Translation System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/translation-agent}
}
```