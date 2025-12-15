# Python Validator Tests - TODO

## Why Tests Are Deferred

The pythonValidator module has its tests deferred because:

1. **Skulpt Initialization Challenge**: Skulpt needs to be loaded and initialized before tests can run. In the browser environment, this happens naturally, but in the test environment (Vitest with jsdom), Skulpt's global initialization doesn't work the same way.

2. **Integration Testing Focus**: The validation logic is tightly integrated with Monaco Editor and the browser environment. Integration tests would provide more value than isolated unit tests.

3. **Manual Testing Coverage**: The validation has been manually tested with:
   - Valid Python syntax → no errors
   - Syntax errors (missing parentheses, etc.) → errors shown
   - Invalid strata_fdtd API usage → warnings shown
   - Error markers in Monaco editor → working
   - Download blocking with errors → working

## Future Test Implementation

To properly test this module, consider:

1. **Mock Skulpt in Tests**: Create a proper Skulpt mock that simulates the parsing behavior
2. **Integration Tests**: Test the full Monaco + validation flow using Playwright or similar
3. **E2E Tests**: Test the complete user workflow in Builder Mode

## Test Scenarios to Cover

When implementing tests, ensure coverage of:

- ✅ Valid Python syntax (no errors)
- ✅ Syntax errors (missing brackets, colons, etc.)
- ✅ Missing imports warnings
- ✅ Invalid grid shape detection
- ✅ Negative resolution detection
- ✅ Undefined variable usage warnings
- ✅ Error vs warning severity distinction
- ✅ Debounced validation behavior
- ✅ Download blocking with errors
- ✅ Performance (<200ms for typical scripts)

## Current Test Command

```bash
# When tests are implemented, run with:
pnpm test src/lib/__tests__/pythonValidator.test.ts
```
