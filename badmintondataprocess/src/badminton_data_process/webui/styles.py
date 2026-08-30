"""Editorial sports-analysis visual system for the local BBA workspace."""

WEBUI_CSS = """
:root {
    --bba-bg: #e9eae5;
    --bba-paper: #f7f6f1;
    --bba-white: #ffffff;
    --bba-ink: #111313;
    --bba-muted: #737a77;
    --bba-line: #d4d7d1;
    --bba-dark: #111515;
    --bba-orange: #f05a32;
    --bba-orange-dark: #cd3e1c;
    --bba-blue: #4368ff;
    --bba-error: #d4473f;
    --bba-radius: 8px;
    --bba-shadow: 0 14px 36px rgba(20, 25, 23, .07);
}
* {box-sizing: border-box;}
body {
    background: var(--bba-bg) !important;
    font-family: Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif !important;
}
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 18px 24px 72px !important;
    color: var(--bba-ink);
}
.bba-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 70px;
    padding: 10px 2px 18px;
    border-bottom: 1px solid #c7cac4;
}
.bba-brand {display: flex; align-items: center; gap: 14px;}
.bba-mark {
    display: grid;
    width: 54px;
    height: 42px;
    place-items: center;
    background: var(--bba-ink);
    color: var(--bba-white);
    font-size: 17px;
    font-weight: 900;
    letter-spacing: -.04em;
    transform: skew(-7deg);
}
.bba-name {display: flex; flex-direction: column; gap: 2px;}
.bba-name b {font-size: 14px; letter-spacing: .01em; color: var(--bba-ink);}
.bba-name small {font-size: 11px; color: var(--bba-muted);}
.runtime-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .13em;
    color: #5e6562;
}
.runtime-badge i {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #24a96b;
    box-shadow: 0 0 0 4px rgba(36,169,107,.11);
}
.bba-intro {
    padding: 74px 0 56px;
    border-bottom: 1px solid #c7cac4;
}
.overline {
    color: var(--bba-orange);
    font-size: 11px;
    font-weight: 850;
    letter-spacing: .17em;
}
.bba-intro h1 {
    max-width: 820px;
    margin: 18px 0 20px;
    font-size: clamp(40px, 6vw, 72px);
    line-height: .98;
    letter-spacing: -.055em;
    color: var(--bba-ink);
}
.bba-intro p {
    max-width: 700px;
    margin: 0;
    font-size: 16px;
    line-height: 1.8;
    color: #5e6562;
}
.stage-rail {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    margin: 0 0 86px;
    background: var(--bba-dark);
    color: var(--bba-white);
}
.stage-node {
    position: relative;
    display: flex;
    align-items: center;
    gap: 12px;
    min-height: 76px;
    padding: 15px 20px;
    border-right: 1px solid rgba(255,255,255,.13);
}
.stage-node:last-child {border-right: 0;}
.stage-node b {color: #868e8a; font-size: 11px; font-weight: 800;}
.stage-node span {display: flex; flex-direction: column; font-size: 13px; font-weight: 750;}
.stage-node small {margin-top: 3px; color: #78807c; font-size: 8px; letter-spacing: .14em;}
.stage-node.is-current::after {
    content: "";
    position: absolute;
    right: 18px;
    bottom: 0;
    left: 18px;
    height: 3px;
    background: var(--bba-orange);
}
.stage-node.is-current b {color: var(--bba-orange);}
.stage-heading {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    margin: 0 0 20px;
    padding-top: 8px;
}
.stage-heading > span {
    display: grid;
    width: 42px;
    height: 42px;
    flex: 0 0 42px;
    place-items: center;
    border: 1px solid #bfc3bd;
    border-radius: 50%;
    color: var(--bba-orange);
    font-size: 12px;
    font-weight: 850;
}
.stage-heading small {color: var(--bba-orange); font-size: 9px; font-weight: 850; letter-spacing: .17em;}
.stage-heading h2 {
    margin: 3px 0 4px;
    color: var(--bba-ink);
    font-size: 28px;
    line-height: 1.15;
    letter-spacing: -.035em;
}
.stage-heading p {margin: 0; color: var(--bba-muted); font-size: 13px;}
.workspace-panel {
    margin-bottom: 78px !important;
    padding: 24px !important;
    border: 1px solid var(--bba-line) !important;
    border-radius: var(--bba-radius) !important;
    background: var(--bba-paper) !important;
    box-shadow: var(--bba-shadow);
}
.video-upload {
    overflow: hidden;
    border: 1px solid #cdd0cb !important;
    border-radius: 5px !important;
    background: #0c0e0e !important;
}
.video-upload video {max-height: 620px; background: #0c0e0e !important;}
.panel-divider {position: relative; margin: 24px 0 18px; border-top: 1px solid var(--bba-line);}
.panel-divider span {
    position: relative;
    top: -10px;
    padding-right: 12px;
    background: var(--bba-paper);
    color: #737a77;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .12em;
}
.config-row {gap: 18px !important;}
.config-row > div {
    padding: 16px !important;
    border: 1px solid var(--bba-line) !important;
    border-radius: 5px !important;
    background: var(--bba-white) !important;
}
.privacy-note {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 18px;
    padding: 13px 15px;
    border-left: 3px solid var(--bba-blue);
    background: #eceffb;
    color: #515a72;
    font-size: 12px;
}
.privacy-note b {color: #273258; white-space: nowrap;}
.calibration-panel {padding: 0 !important; overflow: hidden;}
.calibration-status {
    margin: 0 !important;
    padding: 17px 22px;
    border-bottom: 1px solid var(--bba-line);
    background: #f1f0ea;
    color: #48504c;
    font-size: 13px;
}
.court-preview {
    margin: 0 !important;
    padding: 20px !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: #171a1a !important;
}
.court-preview img {max-height: 720px !important; object-fit: contain !important;}
.primary-actions {gap: 10px !important; padding: 20px 22px; border-top: 1px solid var(--bba-line);}
.primary-actions button, .manual-calibration button {
    min-height: 46px;
    border-radius: 4px !important;
    font-weight: 750 !important;
}
.manual-calibration {
    margin: 0 22px 22px !important;
    border: 1px solid var(--bba-line) !important;
    border-radius: 5px !important;
    background: var(--bba-white) !important;
}
.manual-calibration > .label-wrap {padding: 15px 17px !important; font-weight: 750 !important;}
.manual-help {padding: 4px 3px; color: #626966; font-size: 13px; line-height: 1.7;}
.line-selectors {gap: 8px !important;}
.advanced-corners {margin: 8px 0 !important; border-color: var(--bba-line) !important;}
.launch-panel {
    display: grid !important;
    grid-template-columns: 1fr 310px;
    align-items: center;
    gap: 28px !important;
    margin: -48px 0 82px !important;
    padding: 22px 24px !important;
    border: 0 !important;
    border-radius: var(--bba-radius) !important;
    background: var(--bba-dark) !important;
    color: var(--bba-white);
}
.launch-copy {display: flex; flex-direction: column; gap: 4px;}
.launch-copy small {color: var(--bba-orange); font-size: 9px; font-weight: 850; letter-spacing: .16em;}
.launch-copy b {font-size: 17px;}
.launch-copy span {color: #929a96; font-size: 12px;}
#run-button {
    min-height: 50px;
    border: 0 !important;
    border-radius: 4px !important;
    background: var(--bba-orange) !important;
    color: #fff !important;
    font-size: 14px !important;
    font-weight: 850 !important;
    box-shadow: none !important;
}
#run-button:hover {background: var(--bba-orange-dark) !important;}
#run-button:disabled {background: #3c4240 !important; color: #8d9491 !important;}
.pipeline-progress-card {
    margin: 0 0 12px;
    padding: 26px 28px;
    border: 0;
    border-radius: var(--bba-radius);
    background: var(--bba-dark);
    box-shadow: var(--bba-shadow);
    color: var(--bba-white);
}
.pipeline-progress-heading, .pipeline-progress-detail {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: flex-end;
}
.progress-kicker {display: block; margin-bottom: 5px; color: #727b77; font-size: 9px; font-weight: 850; letter-spacing: .16em;}
.pipeline-progress-heading strong {display: block; color: #fff; font-size: 23px; letter-spacing: -.025em;}
.progress-percent {color: var(--bba-orange); font-size: 38px; line-height: 1; font-weight: 900; letter-spacing: -.04em;}
.pipeline-progress {
    width: 100%;
    height: 6px;
    margin-top: 22px;
    border: 0;
    border-radius: 0;
    accent-color: var(--bba-orange);
    display: block;
}
.pipeline-progress::-webkit-progress-bar {background: #313635;}
.pipeline-progress::-webkit-progress-value {background: var(--bba-orange);}
.progress-segments {display: grid; grid-template-columns: repeat(9,1fr); gap: 5px; margin-top: 9px;}
.progress-segment {height: 3px; background: #313635;}
.progress-segment.is-complete {background: var(--bba-orange);}
.progress-segment.is-active {background: var(--bba-blue);}
.pipeline-progress-detail {margin-top: 18px; color: #8e9692; font-size: 12px;}
.pipeline-progress-detail span {display: flex; flex-direction: column; gap: 3px;}
.pipeline-progress-detail small {color: #626b67; font-size: 9px; text-transform: uppercase; letter-spacing: .1em;}
.pipeline-progress-detail b {color: #c7cdca; font-weight: 650;}
.pipeline-progress-error .progress-percent {color: var(--bba-error);}
.pipeline-progress-error .pipeline-progress {accent-color: var(--bba-error);}
.stage-log {
    margin-bottom: 82px !important;
    border: 1px solid var(--bba-line) !important;
    border-radius: 5px !important;
    background: var(--bba-paper) !important;
}
.stage-log > .label-wrap {
    border-radius: 4px 4px 0 0 !important;
    background: #111313 !important;
    color: #ffffff !important;
}
.stage-log > .label-wrap span,
.stage-log > .label-wrap svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
}
.report-heading {margin-top: 78px;}
.results-tabs {
    border: 1px solid var(--bba-line) !important;
    border-radius: var(--bba-radius) !important;
    background: var(--bba-paper) !important;
    box-shadow: var(--bba-shadow);
    padding: 8px 18px 22px !important;
    overflow: hidden;
}
.results-tabs .tab-nav {gap: 2px; border-bottom: 1px solid var(--bba-line) !important;}
.results-tabs .tab-nav button {
    padding: 14px 16px !important;
    border-radius: 0 !important;
    color: #747b78 !important;
    font-size: 13px;
    font-weight: 700;
}
.results-tabs .tab-nav button.selected {
    border-bottom: 2px solid var(--bba-orange) !important;
    color: var(--bba-ink) !important;
}
.results-video video {max-height: 700px; border-radius: 4px !important; background: #0c0e0e !important;}
.report-summary {padding: 18px 4px;}
.dataframe {border-radius: 4px !important; overflow: hidden;}
button.primary {border-color: var(--bba-orange) !important; background: var(--bba-orange) !important;}
button.primary:hover {background: var(--bba-orange-dark) !important;}

/* High-contrast white theme: keep dark tones only inside media itself. */
body {background: #ffffff !important;}
.gradio-container {
    --body-background-fill: #ffffff;
    --block-background-fill: #ffffff;
    --block-label-background-fill: #ffffff;
    --input-background-fill: #ffffff;
    --body-text-color: #111313;
    --body-text-color-subdued: #555d59;
    --block-label-text-color: #252928;
    --block-info-text-color: #555d59;
    background: #ffffff !important;
}
.bba-intro, .workspace-panel, .results-tabs, .stage-log,
.manual-calibration, .config-row > div {background: #ffffff !important;}
.bba-intro p, .stage-heading p, .manual-help {color: #4f5653 !important;}
.bba-name small, .runtime-badge, .panel-divider span {color: #515855;}
.stage-rail {
    border: 1px solid var(--bba-line);
    background: #ffffff;
    color: var(--bba-ink);
}
.stage-node {border-right-color: var(--bba-line);}
.stage-node b {color: #656c69;}
.stage-node span {color: var(--bba-ink);}
.stage-node small {color: #626966;}
.panel-divider span {background: #ffffff;}
.calibration-status {background: #f6f7f5; color: #343a37;}
.court-preview {background: #f1f2ef !important;}
.launch-panel {
    border: 1px solid var(--bba-line) !important;
    background: #ffffff !important;
    color: var(--bba-ink);
    box-shadow: var(--bba-shadow);
}
.launch-copy b {color: var(--bba-ink);}
.launch-copy span {color: #515855;}
#run-button:disabled {background: #e3e5e1 !important; color: #555d59 !important;}
.pipeline-progress-card {
    border: 1px solid var(--bba-line);
    background: #ffffff;
    color: var(--bba-ink);
}
.pipeline-progress-heading strong {color: var(--bba-ink);}
.progress-kicker {color: #5b625f;}
.pipeline-progress::-webkit-progress-bar, .progress-segment {background: #e2e4e0;}
.pipeline-progress-detail {color: #4f5653;}
.pipeline-progress-detail small {color: #666d69;}
.pipeline-progress-detail b {color: #303634;}
.results-tabs .tab-nav button {color: #535a57 !important;}
.privacy-note {background: #f4f6ff; color: #3f4965;}

/* Text contrast audit: every light surface uses dark text. */
.gradio-container :is(
    label,
    .block-label,
    .block-info,
    .info,
    .prose,
    p,
    li,
    dt,
    dd,
    table,
    th,
    td,
    input,
    textarea,
    select,
    [role="tab"]
) {
    color: #1b211f !important;
}
.gradio-container :is(.block-info, .info, p, li, small) {
    color: #39413e !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {color: #5f6763 !important; opacity: 1;}
.gradio-container .label-wrap,
.gradio-container .label-wrap span {color: #171c1a !important;}
.gradio-container button:not(.primary):not(:disabled) {color: #171c1a !important;}
.gradio-container button.primary,
#run-button {color: #ffffff !important;}
.bba-mark {color: #ffffff !important;}
.stage-log > .label-wrap,
.stage-log > .label-wrap span {
    background: #111313 !important;
    color: #ffffff !important;
}
.stage-log > .label-wrap svg {color: #ffffff !important; stroke: #ffffff !important;}
footer {display: none !important;}
@media (max-width: 820px) {
    .gradio-container {padding: 12px 12px 48px !important;}
    .bba-topbar {align-items: flex-start;}
    .bba-name b {font-size: 12px;}
    .runtime-badge {display: none;}
    .bba-intro {padding: 52px 0 40px;}
    .bba-intro h1 {font-size: 46px;}
    .stage-rail {grid-template-columns: repeat(2,1fr); margin-bottom: 58px;}
    .stage-node:nth-child(2) {border-right: 0;}
    .stage-node:nth-child(-n+2) {border-bottom: 1px solid var(--bba-line);}
    .workspace-panel {padding: 14px !important; margin-bottom: 58px !important;}
    .config-row, .line-selectors {flex-direction: column !important;}
    .launch-panel {grid-template-columns: 1fr; margin-bottom: 60px !important;}
    .stage-log {margin-bottom: 60px !important;}
    .report-heading {margin-top: 58px;}
}
@media (max-width: 520px) {
    .bba-name b {max-width: 210px;}
    .bba-intro h1 {font-size: 38px;}
    .bba-intro p {font-size: 14px;}
    .stage-node {padding: 13px 12px;}
    .stage-heading {gap: 13px;}
    .stage-heading h2 {font-size: 24px;}
    .primary-actions {flex-direction: column !important; padding: 14px;}
    .pipeline-progress-heading, .pipeline-progress-detail {align-items: flex-start; flex-direction: column;}
    .results-tabs {padding: 6px 10px 16px !important;}
}
"""
