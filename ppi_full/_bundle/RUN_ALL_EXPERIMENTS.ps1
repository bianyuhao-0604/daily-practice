# 一键跑完全部实验（顺序：smoke → search → ladder → ablation → baseline → stats → mc）
#
# 用法：
#   .\RUN_ALL_EXPERIMENTS.ps1                             # 完整流程（3 seed，40 trial）
#   .\RUN_ALL_EXPERIMENTS.ps1 -Quick                      # 快速（1 seed，10 trial）
#   .\RUN_ALL_EXPERIMENTS.ps1 -Seeds 42 -Trials 10        # 自定义
#   .\RUN_ALL_EXPERIMENTS.ps1 -SkipSearch                 # 用默认超参，不做搜索
#
# 注意：run_all.py 没有断点续跑，重复执行会覆盖同 tag/seed 的产物。

param(
    [int[]]$Seeds = @(42, 1234, 2024),
    [int]$Bs = 64,
    [double]$Lr = 1e-4,
    [int]$Trials = 40,
    [int]$T = 10,
    [switch]$Quick,
    [switch]$SkipSmoke,
    [switch]$SkipSearch,
    [switch]$SkipBaseline
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if ($Quick) {
    $Seeds = @(42)
    $Trials = 10
}

$py = if (Test-Path -LiteralPath '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' } else { 'python' }
$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$master = Join-Path $logDir "bundle_run_$stamp.log"
$seedArgs = @($Seeds | ForEach-Object { [string]$_ })

$LADDER_TAGS = @('R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7')
$ABL_TAGS = @(
    'L4-unidirectional', 'L4-no-iface-bias', 'L4-no-siface', 'L4-no-surface-gate',
    'L4-no-selfattn-ctx', 'L4-alpha-to-add', 'L4-global-concat-nosym', 'L4-no-global-path',
    'L4-proj-seq-early', 'L5-no-asym', 'L5-no-siface', 'L5-plain-mlp', 'L5-mixup',
    'L5-no-ls', 'L5-no-rdrop', 'L5-concat-asym', 'No-Seq', 'No-SSL'
)

function Invoke-Step {
    param([string]$Title, [string[]]$Argv, [switch]$AllowFail)

    $bar = '=' * 78
    $msg = "[$(Get-Date -Format 'HH:mm:ss')] $bar`n[$(Get-Date -Format 'HH:mm:ss')] $Title`n[$(Get-Date -Format 'HH:mm:ss')] $py $($Argv -join ' ')"
    Write-Host $msg -ForegroundColor Cyan
    Add-Content -LiteralPath $master -Value $msg -Encoding utf8

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $py @Argv 2>&1 | Tee-Object -FilePath $master -Append
    $code = $LASTEXITCODE
    $sw.Stop()

    $done = "[$(Get-Date -Format 'HH:mm:ss')] $Title 结束 exit=$code 用时 $([math]::Round($sw.Elapsed.TotalMinutes,1)) 分钟"
    Write-Host $done -ForegroundColor $(if ($code -eq 0) { 'Green' } else { 'Red' })
    Add-Content -LiteralPath $master -Value $done -Encoding utf8

    if ($code -ne 0 -and -not $AllowFail) {
        throw "$Title 失败（exit=$code），日志：$master"
    }
}

Write-Host "python = $py" -ForegroundColor Yellow
Write-Host "seeds  = $($Seeds -join ', ')   bs=$Bs  lr=$Lr  trials=$Trials" -ForegroundColor Yellow
Write-Host "日志   = $master" -ForegroundColor Yellow

# 1) 冒烟
if (-not $SkipSmoke) {
    Invoke-Step 'smoke（33 项自检）' @('smoke_all.py', '--fresh')
}

# 2) 超参搜索（先做，后面复用 best_hyperparams.json）
$hp = @()
if (-not $SkipSearch) {
    Invoke-Step "search（$Trials trial）" @('run_all.py', 'search', '--trials', "$Trials")
    if (Test-Path -LiteralPath 'workdir\runs\best_hyperparams.json') {
        $hp = @('--hyperparams', 'workdir\runs\best_hyperparams.json')
    }
} elseif (Test-Path -LiteralPath 'workdir\runs\best_hyperparams.json') {
    $hp = @('--hyperparams', 'workdir\runs\best_hyperparams.json')
}

# 3) 阶梯
Invoke-Step 'ladder R0–R7' (@('run_all.py', 'ladder', '--levels') + $LADDER_TAGS + @('--seeds') + $seedArgs + @('--bs', "$Bs", '--lr', "$Lr") + $hp)

# 4) 消融
Invoke-Step 'ablation 18 项' (@('run_all.py', 'ablation', '--names') + $ABL_TAGS + @('--seeds') + $seedArgs + @('--bs', "$Bs", '--lr', "$Lr") + $hp)

# 5) 基线
if (-not $SkipBaseline) {
    Invoke-Step 'baseline RFC-mean / RFC-40' @('run_all.py', 'baseline')
}

# 6) 统计
Invoke-Step 'stats ladder vs R0' (@('run_all.py', 'stats', '--tags') + $LADDER_TAGS + @('--baseline', 'R0'))
Copy-Item -LiteralPath 'workdir\runs\summary.csv' -Destination 'workdir\runs\summary_ladder.csv' -Force
Invoke-Step 'stats ablation vs R7' (@('run_all.py', 'stats', '--tags') + $ABL_TAGS + @('--baseline', 'R7'))
Copy-Item -LiteralPath 'workdir\runs\summary.csv' -Destination 'workdir\runs\summary_ablation.csv' -Force

# 7) MC-Dropout
Invoke-Step 'mc R7' (@('run_all.py', 'mc', '--tags', 'R7', '--seeds') + $seedArgs + @('--T', "$T"))

Write-Host "`n全部完成。汇总：" -ForegroundColor Green
Get-ChildItem -LiteralPath 'workdir\runs' -Filter '*.csv' | Select-Object Name, Length | Format-Table -AutoSize
Write-Host "日志：$master"
