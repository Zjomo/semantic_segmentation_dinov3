`create_hello_vi.py`

- Uses LabVIEW 2021 automation to create `hello.vi`
- Starts from `LabVIEW-MCP-Server-Toolkit/tests/fake_tool_say_hello.vi`
- Rewrites the block-diagram format string from `%s says hello!` to `%s`
- Sets the VI title to `Research MCP Echo`
- Adds MCP-oriented descriptions for the VI, input, and output
- Sets the `input string` default value to `hello world`
- Runs the VI once and saves `resulting string = hello world`

`create_calculator_vi.py`

- Uses LabVIEW 2021 automation to create `calculator.vi`
- Starts from `LabVIEW-MCP-Server-Toolkit/tests/fake tool add.vi`
- Sets the VI title to `Research MCP Calculator`
- Adds MCP-oriented descriptions for the VI, inputs, and output
- Keeps the numeric behavior deterministic as `x + y`
- Sets default values `x = 2`, `y = 3`, and saves `result = 5`

`run_hello_vi.ps1`

- Runs `hello.vi` with LabVIEW 2021
- Default mode uses `-run_then_exit`
- Pass `-KeepOpen` if you want LabVIEW to stay open

`run_calculator_vi.ps1`

- Runs `calculator.vi` with LabVIEW 2021
- Default mode uses `-run_then_exit`
- Pass `-KeepOpen` if you want LabVIEW to stay open

Research intent

- The optimized VI is a deterministic MCP self-test tool for scientific workflows
- It is meant for low-risk validation before acquisition, analysis, or instrument control
- Input and output are intentionally identical so the MCP client can verify round-trip string behavior without side effects
- The companion calculator VI provides deterministic numeric addition for quick scientific sanity checks

Usage

```powershell
python .\create_hello_vi.py
python .\create_calculator_vi.py
.\run_hello_vi.ps1
.\run_calculator_vi.ps1
```
