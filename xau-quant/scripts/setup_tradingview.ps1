# One command setup on Windows: aurum + TradingView MCP + TradingView Desktop + Claude Code.
#
#   cd resq-deck\xau-quant
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tradingview.ps1
#
# Same steps as setup_tradingview.sh. Options: -Yes -Remote -NoLaunch -NoClaude -NoWeb
# Env: TRADINGVIEW_MCP_DIR (default %USERPROFILE%\tradingview-mcp), TV_PORT (default 9222)

param([switch]$Yes, [switch]$Remote, [switch]$NoLaunch, [switch]$NoClaude, [switch]$NoWeb)
$ErrorActionPreference = "Stop"

$Root   = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TvDir  = if ($env:TRADINGVIEW_MCP_DIR) { $env:TRADINGVIEW_MCP_DIR } else { Join-Path $env:USERPROFILE "tradingview-mcp" }
$TvPort = if ($env:TV_PORT) { $env:TV_PORT } else { "9222" }

function Say($m)  { Write-Host "`n==> $m" -ForegroundColor Yellow }
function Ok($m)   { Write-Host "    OK  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    !   $m" -ForegroundColor DarkYellow }
function Die($m)  { Write-Host "    X   $m" -ForegroundColor Red; exit 1 }
function Has($c)  { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

# 1 -- prerequisites
Say "Checking prerequisites"
if (-not (Has git))  { Die "git not found: https://git-scm.com" }
if (-not (Has node)) { Die "Node.js 18+ not found: https://nodejs.org" }
$nodeMajor = [int](node -p "process.versions.node.split('.')[0]")
if ($nodeMajor -lt 18) { Die "Node.js 18+ needed" }
Ok "node $(node --version)"
$Py = if (Has py) { "py" } elseif (Has python) { "python" } else { Die "Python 3.10+ not found: https://python.org" }
& $Py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { Die "Python 3.10+ needed" }
Ok (& $Py --version)
$HaveClaude = Has claude
if (-not $HaveClaude) { Warn "claude CLI not found: npm install -g @anthropic-ai/claude-code" }

# 2 -- TradingView MCP
Say "TradingView MCP in $TvDir"
if (Test-Path (Join-Path $TvDir ".git")) { git -C $TvDir pull --ff-only -q; Ok "updated" }
else { git clone -q https://github.com/tradesdontlie/tradingview-mcp.git $TvDir; Ok "cloned" }
Push-Location $TvDir; npm install --no-audit --no-fund --silent; Pop-Location; Ok "npm install"

# 3 -- aurum
Say "aurum Python environment in $Root\.venv"
$VPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) { & $Py -m venv (Join-Path $Root ".venv") }
& $VPy -m pip install -q --upgrade pip
& $VPy -m pip install -q -e "$Root[mcp,llm,data]"
Ok "aurum $(& $VPy -c 'import aurum; print(aurum.__version__)') installed"

# 4 -- compile on TradingView's server
Say "Compiling tradingview\aurum_gold.pine on TradingView's server"
$check = node (Join-Path $TvDir "src\cli\index.js") pine check --file (Join-Path $Root "tradingview\aurum_gold.pine") 2>&1 | Out-String
Write-Host ($check.Trim() -split "`n" | Select-Object -First 40 | ForEach-Object { "    $_" } | Out-String)
if ($check -match '"compiled":\s*true') { Ok "compiles" } else { Warn "not confirmed; the Claude run will compile on the chart" }

# 5 -- register MCP servers (local scope, this folder)
if ($HaveClaude) {
  Say "Registering MCP servers with Claude Code"
  Push-Location $Root
  foreach ($n in "aurum", "tradingview") { claude mcp remove -s local $n 2>$null | Out-Null }
  claude mcp add -s local aurum -- $VPy -m aurum.mcp_server | Out-Null; Ok "aurum"
  claude mcp add -s local tradingview -- node (Join-Path $TvDir "src\server.js") | Out-Null; Ok "tradingview"
  Pop-Location
}

# 6 -- dashboard
if (-not $NoWeb) {
  Say "Starting the aurum dashboard"
  $up = $false
  try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:8765/api/strategies" | Out-Null; $up = $true } catch {}
  if (-not $up) {
    New-Item -ItemType Directory -Force (Join-Path $env:USERPROFILE ".aurum") | Out-Null
    Start-Process -WindowStyle Hidden -FilePath $VPy -ArgumentList "-m", "aurum.web", "--port", "8765" -WorkingDirectory $Root
    Start-Sleep 2
  }
  Ok "http://127.0.0.1:8765"
  Start-Process "http://127.0.0.1:8765"
}

# 7 -- TradingView Desktop with the debug port
if (-not $NoLaunch) {
  Say "TradingView Desktop with --remote-debugging-port=$TvPort"
  $listening = $false
  try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$TvPort/json/version" | Out-Null; $listening = $true } catch {}
  if ($listening) { Ok "already listening on $TvPort" }
  else {
    $go = $Yes
    if (-not $go) { $a = Read-Host "    This quits TradingView and reopens it in debug mode. Continue? [Y/n]"; $go = ($a -eq "" -or $a -match "^[Yy]") }
    if ($go) { cmd /c "`"$(Join-Path $TvDir 'scripts\launch_tv_debug.bat')`" $TvPort" }
    else { Warn "skipped; start TradingView with --remote-debugging-port=$TvPort yourself" }
  }
}

# 8 -- hand over to Claude Code
if (-not $NoClaude -and $HaveClaude) {
  Set-Location $Root
  if ($Remote) {
    Say "Starting Claude Code Remote Control. In the Claude app, open this session and send: /tradingview-run"
    claude remote-control
    exit
  }
  Say "Opening Claude Code with the run playbook (approve the MCP servers if asked)"
  $playbook = Get-Content -Raw (Join-Path $Root ".claude\commands\tradingview-run.md")
  claude $playbook
  exit
}
Say "Done. Run 'claude' in $Root and type /tradingview-run"
