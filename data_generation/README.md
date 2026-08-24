# Trait Artifact Generation

This directory contains tooling for generating trait artifacts (instructions, questions, and evaluation prompts) using Claude Sonnet with extended thinking.

## Usage

`generate_trait_artifacts.py` is the command-line tool for generating trait artifacts.

```bash
# Generate a single predefined trait
python generate_trait_artifacts.py --trait evil --output both

# Generate all predefined traits (evil, sycophantic, hallucinating, etc.)
python generate_trait_artifacts.py --all

# Generate a custom trait
python generate_trait_artifacts.py \
    --trait verbose \
    --description "Responses that are excessively long and detailed" \
    --question-instruction "Focus on questions that could be answered concisely" \
    --output both

# Validate existing artifacts
python generate_trait_artifacts.py --validate-only

# Quiet mode (less verbose output)
python generate_trait_artifacts.py --trait evil --quiet
```

**Arguments:**
- `--trait`: Name of the trait to generate
- `--description`: Description of the trait (for custom traits)
- `--question-instruction`: Additional instructions for question generation
- `--output`: Where to save (`extract`, `eval`, or `both`)
- `--all`: Generate all predefined traits
- `--validate-only`: Only validate existing artifacts
- `--quiet`: Suppress verbose output

## Prerequisites

### Environment Variables

Set in your `.env` file:

```bash
ANTHROPIC_API_KEY=your_key_here  # Claude API key
# OR
OPENAI_API_KEY=your_key_here     # If using unified API
```

The script will try `ANTHROPIC_API_KEY` first, then fall back to `OPENAI_API_KEY`.

### Dependencies

All dependencies are included in the main `requirements.txt`:
- `anthropic==0.40.0` (for Claude API)

## Output Structure

Generated artifacts are saved as JSON files in:
- `trait_data_extract/`: For extraction set (used to generate persona vectors)
- `trait_data_eval/`: For evaluation set (used to evaluate models)

Each artifact contains:

```json
{
  "instruction": [
    {"pos": "positive instruction 1", "neg": "negative instruction 1"},
    {"pos": "positive instruction 2", "neg": "negative instruction 2"},
    ...  // 5 pairs total
  ],
  "questions": [
    "question 1",
    "question 2",
    ...  // 40 questions total
  ],
  "eval_prompt": "Evaluation prompt with {question} and {answer} placeholders"
}
```

## Predefined Traits

The following 13 traits are predefined and can be generated with `--all`.

Generic traits inherited from upstream:

- **evil**: Malicious intent, seeking to harm or cause suffering
- **sycophantic**: Excessively agreeable, prioritizing flattery over truth
- **hallucinating**: Confidently stating false information
- **impolite**: Rude, disrespectful, or harsh in tone
- **humorous**: Incorporating jokes, wit, or comedic elements
- **apathetic**: Showing lack of interest, emotion, or concern
- **optimistic**: Emphasizing positive outcomes and hopeful perspectives

This project's own research traits, used by `scripts/batch_generate_vec.sh`:

- **altruism**: Benefitting others at a cost or risk to oneself, expecting nothing back
- **forgiveness**: Giving second chances and restoring trust after it was broken
- **retaliation**: Punishing wrongdoing so that harmful actions have consequences
- **conditional_cooperation**: Matching behavior to how others have acted before
- **expected_altruism**: Predicting that others will be generous rather than self-interested
- **expected_forgiveness**: Predicting that others will forgive rather than hold a grudge

## Model Configuration

The generation uses:
- **Model**: `claude-sonnet-4-5-20250929` (Claude Sonnet 4.5)
- **Extended thinking**: Enabled with 5000 token budget
- **Max tokens**: 16000

This configuration matches the original paper's specifications.

## Validation

Artifacts are automatically validated to ensure:
- 5 instruction pairs (each with `pos` and `neg`)
- 40 questions
- Evaluation prompt with `{question}` and `{answer}` placeholders

Run validation on existing artifacts:

```bash
python generate_trait_artifacts.py --validate-only
```

## Examples

### Example 1: Generate a Single Trait

```bash
python generate_trait_artifacts.py --trait evil
```

### Example 2: Generate Custom Trait

```bash
python generate_trait_artifacts.py \
    --trait creative \
    --description "Responses that demonstrate original and imaginative thinking" \
    --question-instruction "Ask open-ended questions that allow for creative responses"
```

### Example 3: Batch Generate All Traits

```bash
python generate_trait_artifacts.py --all
```

This will generate artifacts for all 13 predefined traits and save them to both extract and eval directories.

## Troubleshooting

### API Key Not Found

```
Error: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable not set
```

**Solution**: Add your API key to the `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### JSON Parsing Error

If the model returns malformed JSON, the script will raise an error. This is usually a temporary issue - try running again.

### Validation Failures

If an artifact fails validation, check:
- Correct number of instruction pairs (5)
- Correct number of questions (40)
- Eval prompt contains `{question}` and `{answer}` placeholders

## Prompts

The generation prompts are defined in `prompts.py`. The main prompt template (`PROMPTS["generate_trait"]`) instructs Claude to:

1. Generate 5 instruction pairs (positive and negative)
2. Design 40 diverse questions
3. Create an evaluation prompt for scoring trait presence

You can modify `prompts.py` to customize the generation behavior.
